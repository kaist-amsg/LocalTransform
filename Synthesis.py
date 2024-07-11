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
from scripts.get_edit import get_bg_partition, combined_edit
from LocalTemplate.template_collector import Collector

atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
             'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
             'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
             'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir',
             'Ce', 'Gd', 'Ga', 'Cs']

def demap(smiles):
    mol = Chem.MolFromSmiles(smiles)
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
    return Chem.MolToSmiles(mol)

class localtransform():
    def __init__(self, dataset, device='cuda:0'):
        self.data_dir = 'data/%s' % dataset
        self.config_path = 'data/configs/default_config'
        self.model_path = 'models/LocalTransform_%s.pth' % dataset
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.args = {'data_dir': self.data_dir, 'model_path': self.model_path, 'config_path': self.config_path, 'device': self.device, 'mode': 'test'}
        self.template_dicts, self.template_infos = self.load_templates()
        self.model, self.graph_function = self.init_model()
    
    def load_templates(self):
        template_dicts = {}
        for site in ['real', 'virtual']:
            template_df = pd.read_csv('%s/%s_templates.csv' % (self.data_dir, site))
            template_dict = {template_df['Class'][i]: template_df['Template'][i].split('_') for i in template_df.index}
            print ('loaded %s %s templates' % (len(template_dict), site))
            template_dicts[site[0]] = template_dict
        template_infos = pd.read_csv('%s/template_infos.csv' % self.data_dir)
        template_infos = {template_infos['Template'][i]: {
            'edit_site': eval(template_infos['edit_site'][i]),
            'change_H': eval(template_infos['change_H'][i]), 
            'change_C': eval(template_infos['change_C'][i]), 
            'change_S': eval(template_infos['change_S'][i])} for i in template_infos.index}
        return template_dicts, template_infos

    def init_model(self):
        self.args = init_featurizer(self.args)
        model = load_model(self.args)
        model.eval()
        smiles_to_graph = partial(smiles_to_bigraph, add_self_loop=True)
        node_featurizer = WeaveAtomFeaturizer(atom_types=atom_types)
        edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
        graph_function = lambda s: smiles_to_graph(s, node_featurizer = node_featurizer, edge_featurizer = edge_featurizer, canonical_atom_order = False)
        return model, graph_function

    def make_inference(self, reactant_list, topk=5):
        fgraphs = []
        dgraphs = []
        for smiles in reactant_list:
            mol = Chem.MolFromSmiles(smiles)
            fgraph = self.graph_function(smiles)
            dgraph = {'atom_distance_matrix': get_adm(mol), 'bonds':get_bonds(smiles)}
            dgraph['v_bonds'], dgraph['r_bonds'] = dgraph['bonds']
            fgraphs.append(fgraph)
            dgraphs.append(dgraph)
        bg = dgl.batch(fgraphs)
        bg.set_n_initializer(dgl.init.zero_initializer)
        bg.set_e_initializer(dgl.init.zero_initializer)
        adm_lists = [graph['atom_distance_matrix'] for graph in dgraphs]
        adms = pad_atom_distance_matrix(adm_lists)
        bonds_dicts = {'virtual': [torch.from_numpy(graph['v_bonds']).long() for graph in dgraphs], 'real': [torch.from_numpy(graph['r_bonds']).long() for graph in dgraphs]}

        with torch.no_grad():
            pred_VT, pred_RT, _, _, pred_VI, pred_RI, attentions = predict(self.args, self.model, bg, adms, bonds_dicts)
            pred_VT = nn.Softmax(dim=1)(pred_VT)
            pred_RT = nn.Softmax(dim=1)(pred_RT)
            v_sep, r_sep = get_bg_partition(bg, bonds_dicts)
            start_v, start_r = 0, 0
            predictions = []
            for i, (reactant) in enumerate(reactant_list):
                end_v, end_r = v_sep[i], r_sep[i]
                virtual_bonds, real_bonds = bonds_dicts['virtual'][i].numpy(), bonds_dicts['real'][i].numpy()
                pred_vi, pred_ri = pred_VI[i].cpu(), pred_RI[i].cpu()
                pred_v, pred_r = pred_VT[start_v:end_v], pred_RT[start_r:end_r]
                prediction = combined_edit(virtual_bonds, real_bonds, pred_vi, pred_ri, pred_v, pred_r, topk*10)
                predictions.append(prediction)
                start_v = end_v
                start_r = end_r
        return predictions
            
    def predict_product(self, reactant_list, topk=5, verbose=0):
        if isinstance(reactant_list, str):
            reactant_list = [reactant_list]
        predictions = self.make_inference(reactant_list, topk)
        results_df = {'Reactants' : []}
        results_dict = {}
        for k in range(topk):
            results_df['Top-%d' % (k+1)] = []
        
        for reactant, prediction in zip(reactant_list, predictions):
            pred_types, pred_sites, scores = prediction
            collector = Collector(reactant, self.template_infos, 'nan', False, verbose = verbose > 1)
            for k, (pred_type, pred_site, score) in enumerate(zip(pred_types, pred_sites, scores)):
                template, H_code, C_code, S_code, action = self.template_dicts[pred_type][pred_site[1]]
                pred_site = pred_site[0]
                if verbose > 0:
                    print ('%dth prediction:' % (k+1), template, action, pred_site, score)
                collector.collect(template, H_code, C_code, S_code, action, pred_site, score)
                if len(collector.predictions) >= topk:
                    break
            sorted_predictions = [k for k, v in sorted(collector.predictions.items(), key=lambda item: -item[1]['score'])]
            results_df['Reactants'].append(Chem.MolFromSmiles(reactant))
            results_dict[reactant] = {}
            for k in range(topk):
                p = sorted_predictions[k] if len(sorted_predictions)>k else ''
                results_dict[reactant]['Top-%d' % (k+1)] = collector.predictions[p]
                results_dict[reactant]['Top-%d' % (k+1)]['product'] = p
                results_df['Top-%d' % (k+1)].append(Chem.MolFromSmiles(p))
                
        results_df = pd.DataFrame(results_df)
        return results_df, results_dict

#     def predict_products(self, args, reactant_list, model, graph_functions, template_dicts, template_infos, product = None, reagents = 'nan', top_k = 5, collect_n = 100, verbose = 0, sep = False):
#         model.eval()
#         if reagents != 'nan':
#             smiles = reactant + '.' + reagents
#         else:
#             smiles = reactant
#         dglgraph = graph_functions(smiles)
#         adms = pad_atom_distance_matrix([get_adm(Chem.MolFromSmiles(smiles))])
#         v_bonds, r_bonds = get_bonds(smiles)
#         bonds_dicts = {'virtual': [torch.from_numpy(v_bonds).long()],  'real': [torch.from_numpy(r_bonds).long()]}
#         with torch.no_grad():
#             pred_VT, pred_RT, _, _, pred_VI, pred_RI, attentions = predict(args, model, dglgraph, adms, bonds_dicts)
#             pred_v = nn.Softmax(dim=1)(pred_VT)
#             pred_r = nn.Softmax(dim=1)(pred_RT)
#             pred_vi = pred_VI[0].cpu()
#             pred_ri = pred_RI[0].cpu()
#             pred_types, pred_sites, pred_scores = combined_edit(v_bonds, r_bonds, pred_vi, pred_ri, pred_v, pred_r, collect_n)

#         collector = Collector(reactant, template_infos, reagents, sep, verbose = verbose > 1)
#         for k, (pred_type, pred_site, score) in enumerate(zip(pred_types, pred_sites, pred_scores)):
#             template, H_code, C_code, S_code, action = template_dicts[pred_type][pred_site[1]]
#             pred_site = pred_site[0]
#             if verbose > 0:
#                 print ('%sth prediction:' % k, template, action, pred_site, score)
#             collector.collect(template, H_code, C_code, S_code, action, pred_site, score)
#             if len(collector.predictions) >= top_k:
#                 break    
#         sort_predictions = [k for k, v in sorted(collector.predictions.items(), key=lambda item: -item[1]['score'])]

#         reactant = demap(reactant)
#         if product != None:
#             correct_at = False
#             product = demap(product)

#         results_dict = {'Reactants' : demap(reactant)}
#         results_df = pd.DataFrame({'Reactants' : [Chem.MolFromSmiles(reactant)]})
#         for k, p in enumerate(sort_predictions):
#             results_dict['Top-%d' % (k+1)] = collector.predictions[p]
#             results_dict['Top-%d' % (k+1)]['product'] = p
#             results_df['Top-%d' % (k+1)] = [Chem.MolFromSmiles(p)]
#             if product != None:
#                 if set(p.split('.')).intersection(set(product.split('.'))):
#                     correct_at = k+1

#         if product != None:
#             results_df['Correct at'] = correct_at
#         return results_df, results_dict
