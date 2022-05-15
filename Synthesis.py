from itertools import permutations
import pandas as pd
import json
from rdkit import Chem 

import torch
from torch import nn
import sklearn

import dgl
from dgllife.utils import smiles_to_bigraph, WeaveAtomFeaturizer, CanonicalBondFeaturizer
from functools import partial

from scripts.dataset import combine_reactants, get_bonds, get_adm
from scripts.utils import init_featurizer, load_model, pad_atom_distance_matrix, predict
from scripts.get_edit import combined_edit
from LocalTemplate.template_collector import Collector

def load_templates(args):
    template_dicts = {}
    for site in ['real', 'virtual']:
        template_df = pd.read_csv('%s/%s_templates.csv' % (args['data_dir'], site))
        template_dict = {template_df['Class'][i]: template_df['Template'][i].split('_') for i in template_df.index}
        print ('loaded %s %s templates' % (len(template_dict), site))
        template_dicts[site[0]] = template_dict
    template_infos = pd.read_csv('%s/template_infos.csv' % args['data_dir'])
    template_infos = {template_infos['Template'][i]: {'edit_site': eval(template_infos['edit_site'][i]), 'change_H': eval(template_infos['change_H'][i]), 'change_C': eval(template_infos['change_C'][i]), 'change_S': eval(template_infos['change_S'][i])} for i in template_infos.index}

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
    edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
    graph_function = lambda s: smiles_to_graph(s, node_featurizer = node_featurizer, edge_featurizer = edge_featurizer, canonical_atom_order = False)
    return model, graph_function, template_dicts, template_infos

def demap(smiles):
    mol = Chem.MolFromSmiles(smiles)
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
    return Chem.MolToSmiles(mol)

def predict_product(args, reactant, model, graph_functions, template_dicts, template_infos, product = None, reagents = 'nan', top_k = 5, collect_n = 100, verbose = 0, sep = False):
    model.eval()
    if reagents != 'nan':
        smiles = reactant + '.' + reagents
    else:
        smiles = reactant
    dglgraph = graph_functions(smiles)
    adms = pad_atom_distance_matrix([get_adm(Chem.MolFromSmiles(smiles))])
    v_bonds, r_bonds = get_bonds(smiles)
    bonds_dicts = {'virtual': [torch.from_numpy(v_bonds).long()],  'real': [torch.from_numpy(r_bonds).long()]}
    with torch.no_grad():
        pred_VT, pred_RT, _, _, pred_VI, pred_RI, attentions = predict(args, model, dglgraph, adms, bonds_dicts)
        pred_v = nn.Softmax(dim=1)(pred_VT)
        pred_r = nn.Softmax(dim=1)(pred_RT)
        pred_vi = pred_VI[0].cpu()
        pred_ri = pred_RI[0].cpu()
        pred_types, pred_sites, pred_scores = combined_edit(v_bonds, r_bonds, pred_vi, pred_ri, pred_v, pred_r, collect_n)

    collector = Collector(reactant, template_infos, reagents, sep, verbose = verbose > 1)
    for k, (pred_type, pred_site, score) in enumerate(zip(pred_types, pred_sites, pred_scores)):
        template, H_code, C_code, S_code, action = template_dicts[pred_type][pred_site[1]]
        pred_site = pred_site[0]
        if verbose > 0:
            print ('%sth prediction:' % k, template, action, pred_site, score)
        collector.collect(template, H_code, C_code, S_code, action, pred_site, score)
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