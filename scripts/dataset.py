import os, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem

import torch
import sklearn
import dgl
import dgl.backend as F
from dgl.data.utils import save_graphs, load_graphs

def combine_reactants(reactant, reagent):
    if str(reagent) == 'nan':
        smiles = reactant
    else:
        smiles = '.'.join([reactant, reagent])
    return smiles

def get_bonds(smiles):
    mol = Chem.MolFromSmiles(smiles)
    A = [a for a in range(mol.GetNumAtoms())]
    B = []
    for atom in mol.GetAtoms():
        others = []
        bonds = atom.GetBonds()
        for bond in bonds:
            atoms = [bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()]
            other = [a for a in atoms if a != atom.GetIdx()][0]
            others.append(other)
        b = [(atom.GetIdx(), other) for other in sorted(others)]
        B += b
    V = []
    for a in A:
        V += [(a,b) for b in A if a != b and (a,b) not in B]
    return np.array(V), np.array(B)
    
def get_adm(mol, max_distance = 6):
    mol_size = mol.GetNumAtoms()
    distance_matrix = np.ones((mol_size, mol_size)) * max_distance + 1
    dm = Chem.GetDistanceMatrix(mol)
    dm[dm > 100] = -1 # remote (different molecule)
    dm[dm > max_distance] = max_distance # remote (same molecule)
    dm[dm == -1] = max_distance + 1
    distance_matrix[:dm.shape[0],:dm.shape[1]] = dm
    return distance_matrix
    
class USPTODataset(object):
    def __init__(self, args, mol_to_graph, node_featurizer, edge_featurizer, load=True, log_every=1000):
        df = pd.read_csv('%s/labeled_data.csv' % args['data_dir'])
        self.train_ids = df.index[df['Split'] == 'train'].values
        self.val_ids = df.index[df['Split'] == 'valid'].values
        self.test_ids = df.index[df['Split'] == 'test'].values
        self.reactants = df['Reactants'].tolist()
        self.reagents = df['Reagents'].tolist()
        self.masks = df['Mask'].tolist()
        
        self.sep = args['sep']
        self.fgraph_path = '../data/saved_graphs/full_%s_fgraph.bin' % args['dataset']
        if self.sep:
            self.labels = [eval(t) for t in df['Labels_sep']]
            self.dgraph_path = '../data/saved_graphs/full_%s_dgraph_sep.pkl' % args['dataset']  # changed by numpy matrix for faster processing speed
        else:
            self.labels = [eval(t) for t in df['Labels_mix']]
            self.dgraph_path = '../data/saved_graphs/full_%s_dgraph_mix.pkl' % args['dataset']
        self._pre_process(mol_to_graph, node_featurizer, edge_featurizer)
    
    def _pre_process(self, mol_to_graph, node_featurizer, edge_featurizer):
        self.fgraphs_exist = os.path.exists(self.fgraph_path)
        self.dgraphs_exist = os.path.exists(self.dgraph_path)
        self.fgraphs = []
        self.dgraphs = []
                    
        if not self.fgraphs_exist or not self.dgraphs_exist:   
            for (s1, s2) in tqdm(zip(self.reactants, self.reagents), total=len(self.reactants), desc='Building dgl graphs...'):
                smiles = combine_reactants(s1, s2)  # s1.s2
                mol = Chem.MolFromSmiles(smiles)
                if not self.fgraphs_exist:
                    fgraph = mol_to_graph(mol, node_featurizer=node_featurizer, edge_featurizer=edge_featurizer, canonical_atom_order=False)
                    self.fgraphs.append(fgraph)
                if  not self.dgraphs_exist:
                    if self.sep:
                        self.dgraphs.append({'atom_distance_matrix': get_adm(mol), 'bonds':get_bonds(s1)})
                    else:
                        self.dgraphs.append({'atom_distance_matrix': get_adm(mol), 'bonds':get_bonds(smiles)})
                        
        if self.fgraphs_exist:
            print ('Loading feture graphs from %s...' % self.fgraph_path)
            self.fgraphs, _ = load_graphs(self.fgraph_path)
        else:
            save_graphs(self.fgraph_path, self.fgraphs)
                
        if self.dgraphs_exist:
            print ('Loading dense graphs from %s...' % self.dgraph_path)
            with open(self.dgraph_path, 'rb') as f:
                self.dgraphs = pickle.load(f)
        else:
            with open(self.dgraph_path, 'wb') as f:
                pickle.dump(self.dgraphs, f)
            
    def __getitem__(self, item):
        dgraph = self.dgraphs[item]
        dgraph['v_bonds'], dgraph['r_bonds'] = dgraph['bonds']
        return self.reactants[item], self.reagents[item], self.fgraphs[item], dgraph, self.labels[item], self.masks[item]

    def __len__(self):
            return len(self.reactants)
        
class USPTOTestDataset(object):
    def __init__(self, args, mol_to_graph, node_featurizer, edge_featurizer, load=True, log_every=1000):
        df = pd.read_csv('%s/preprocessed_test.csv' % args['data_dir'])
        self.reactants = df['Reactants'].tolist()
        self.reagents = df['Reagents'].tolist()
        self.sep = args['sep']
        self.fgraph_path = '../data/saved_graphs/test_%s_fgraph.bin' % args['dataset']
        if self.sep:
            self.dgraph_path = '../data/saved_graphs/test_%s_dgraph_sep.pkl' % args['dataset']  # changed by numpy matrix for faster processing speed
        else:
            self.dgraph_path = '../data/saved_graphs/test_%s_dgraph_mix.pkl' % args['dataset']
        self._pre_process(mol_to_graph, node_featurizer, edge_featurizer)
        
    
    def _pre_process(self, mol_to_graph, node_featurizer, edge_featurizer):
        self.fgraphs_exist = os.path.exists(self.fgraph_path)
        self.dgraphs_exist = os.path.exists(self.dgraph_path)
        self.fgraphs = []
        self.dgraphs = []
                    
        if not self.fgraphs_exist or not self.dgraphs_exist:   
            for (s1, s2) in tqdm(zip(self.reactants, self.reagents), total=len(self.reactants), desc='Building dgl graphs...'):
                smiles = combine_reactants(s1, s2)  # s1.s2
                if not self.fgraphs_exist:
                    mol = Chem.MolFromSmiles(smiles)
                    fgraph = mol_to_graph(mol, node_featurizer=node_featurizer, edge_featurizer=edge_featurizer, canonical_atom_order=False)
                    self.fgraphs.append(fgraph)
                if  not self.dgraphs_exist:
                    if self.sep:
                        self.dgraphs.append({'atom_distance_matrix': get_adm(mol), 'bonds':get_bonds(s1)})
                    else:
                        self.dgraphs.append({'atom_distance_matrix': get_adm(mol), 'bonds':get_bonds(smiles)})
                        
        if self.fgraphs_exist:
            print ('Loading feture graphs from %s...' % self.fgraph_path)
            self.fgraphs, _ = load_graphs(self.fgraph_path)
        else:
            save_graphs(self.fgraph_path, self.fgraphs)
                
        if self.dgraphs_exist:
            print ('Loading dense graphs from %s...' % self.dgraph_path)
            with open(self.dgraph_path, 'rb') as f:
                self.dgraphs = pickle.load(f)
        else:
            with open(self.dgraph_path, 'wb') as f:
                pickle.dump(self.dgraphs, f)
            
    def __getitem__(self, item):
        dgraph = self.dgraphs[item]
        dgraph['v_bonds'], dgraph['r_bonds'] = dgraph['bonds']
        return self.reactants[item], self.reagents[item], self.fgraphs[item], dgraph

    def __len__(self):
            return len(self.reactants)
        
