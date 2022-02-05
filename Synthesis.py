from itertools import permutations
import pandas as pd
import json

import torch
from torch import nn
import sklearn

from rdkit import Chem 

import dgl
from dgllife.utils import smiles_to_bigraph, WeaveAtomFeaturizer, CanonicalBondFeaturizer
from functools import partial

from script.utils import init_featurizer, load_model, atom_position_matrix
from script.get_edit import combined_edit
from LocalTemplate.template_collector import Collector

def make_densegraph(s, smiles_to_graph):
    g = smiles_to_graph(s, canonical_atom_order=False)
    node_n = g.number_of_nodes()
    no_bonds = []
    real_bonds = [tuple(b) for b in torch.transpose(g.adjacency_matrix().coalesce().indices(), 0, 1) if b[0] != b[1]]
    virtual_bonds = list(permutations(list(range(node_n)), 2))
    for bond in real_bonds:
        virtual_bonds.remove(bond)
    if len(real_bonds) == 0:
        real_bonds = [(0,0)]
        no_bonds.append('real')
    if len(virtual_bonds) == 0:
        virtual_bonds = [(0,0)]
        no_bonds.append('virtual')
    hg = dgl.heterograph({
        ('atom', 'real', 'atom'): (torch.tensor(real_bonds)[:,0], torch.tensor(real_bonds)[:,1]),
        ('atom', 'virtual', 'atom'): (torch.tensor(virtual_bonds)[:,0], torch.tensor(virtual_bonds)[:,1])})
    for etype in no_bonds:
        hg.remove_edges([0, 0], etype)
    return hg
    

def predict(model, rpms, graph, hgraph, device):
    device = torch.device(device)
    bg = dgl.batch([graph])
    hbg = dgl.batch([hgraph])
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    bg = bg.to(device)
    hbg = hbg.to(device)
    node_feats = bg.ndata.pop('h').to(device)
    edge_feats = bg.edata.pop('e').to(device)
    return model(rpms, bg, hbg, node_feats, edge_feats)

def load_templates(args):
    template_dicts = {}
    for site in ['real', 'virtual']:
        template_df = pd.read_csv('%s/%s_templates.csv' % (args['data_dir'], site))
        template_dict = {template_df['Class'][i]: template_df['Template'][i].split('_') for i in template_df.index}
        print ('loaded %s %s templates' % (len(template_dict), site))
        template_dicts[site[0]] = template_dict
    template_infos = pd.read_csv('%s/template_infos.csv' % args['data_dir'])
    template_infos = {template_infos['Template'][i]: {'edit_site': eval(template_infos['edit_site'][i]), 'change_H': eval(template_infos['change_H'][i]), 'change_C': eval(template_infos['change_C'][i])} for i in template_infos.index}

    return template_dicts, template_infos

def init_LocalTransform(args):
    args = init_featurizer(args)
    model = load_model(args)
    template_dicts, template_infos = load_templates(args)
    smiles_to_graph = partial(smiles_to_bigraph, add_self_loop=True)
    atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
             'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
             'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
             'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir',
             'Ce', 'Gd', 'Ga', 'Cs']
    node_featurizer = WeaveAtomFeaturizer(atom_types = atom_types)
#     node_featurizer = NCAtomFeaturizer()
    edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
    graph_function1 = lambda s: smiles_to_graph(s, node_featurizer = node_featurizer, edge_featurizer = edge_featurizer, canonical_atom_order = False)
    graph_function2 = lambda s: make_densegraph(s, smiles_to_graph)
    return model, (graph_function1, graph_function2), template_dicts, template_infos

def demap(smiles):
    mol = Chem.MolFromSmiles(smiles)
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
    return Chem.MolToSmiles(mol)

def predict_product(reactant, model, graph_functions, device, template_dicts, template_infos, product = None, reagents = 'nan', top_k = 5, collect_n = 100, verbose = 0, sep = False):
    model.eval()
    if reagents != 'nan':
        smiles = reactant + '.' + reagents
    else:
        smiles = reactant
    rpms = atom_position_matrix([reactant], [reagents])
    graph, hgraph = graph_functions[0](smiles), graph_functions[1](reactant)
    with torch.no_grad():
        pred_VT, pred_RT, _, _, pred_VI, pred_RI, attentions = predict(model, rpms, graph, hgraph, device)
        pred_v = nn.Softmax(dim=1)(pred_VT)
        pred_r = nn.Softmax(dim=1)(pred_RT)
        pred_vi = pred_VI[0].cpu()
        pred_ri = pred_RI[0].cpu()
        pred_types, pred_sites, pred_scores = combined_edit(hgraph, pred_vi, pred_ri, pred_v, pred_r, collect_n)

    collector = Collector(reactant, template_infos, reagents, sep, verbose = verbose > 1)
    for k, (pred_type, pred_site, score) in enumerate(zip(pred_types, pred_sites, pred_scores)):
        template, H_code, C_code, action = template_dicts[pred_type][pred_site[1]]
        pred_site = pred_site[0]
        if verbose > 0:
            print ('%sth prediction:' % k, template, action, pred_site, score)
        collector.collect(template, H_code, C_code, action, pred_site, score)
        if len(collector.predictions) >= top_k:
            break    
    sort_predictions = [k for k, v in sorted(collector.predictions.items(), key=lambda item: -item[1]['score'])]
    
    reactant = demap(reactant)
    if product != None:
        correct_at = False
        product = demap(product)
        
    results_dict = {'Reactants' : demap(reactant)}
    results_df = pd.DataFrame({'Reactants' : [Chem.MolFromSmiles(reactant)]})
    for k, p in enumerate(sort_predictions):
        results_dict['Top-%d' % (k+1)] = collector.predictions[p]
        results_dict['Top-%d' % (k+1)]['product'] = p
        results_df['Top-%d' % (k+1)] = [Chem.MolFromSmiles(p)]
        if product != None:
            if set(p.split('.')).intersection(set(product.split('.'))):
                correct_at = k+1
            
    if product != None:
        results_df['Correct at'] = correct_at
    return results_df, results_dict

