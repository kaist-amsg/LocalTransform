import os
import pandas as pd
import multiprocessing
from tqdm import tqdm
from itertools import permutations

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

def make_densegraph(s, mol_to_graph):
    m = Chem.MolFromSmiles(s)
    g = mol_to_graph(m, canonical_atom_order=False)
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
    dg = dgl.heterograph({
        ('atom', 'real', 'atom'): (torch.tensor(real_bonds)[:,0], torch.tensor(real_bonds)[:,1]),
        ('atom', 'virtual', 'atom'): (torch.tensor(virtual_bonds)[:,0], torch.tensor(virtual_bonds)[:,1])})
    for etype in no_bonds:
        dg.remove_edges([0, 0], etype)
        
    if '_ID' in dg.ndata:
        dg.ndata.pop('_ID')
        dg.edata.pop('_ID')
        
    return dg
    
class USPTODataset(object):
    def __init__(self, args, mol_to_graph, node_featurizer, edge_featurizer, load=True, log_every=1000):
#         if os.path.isfile('%s/relabeled_data.csv' % args['data_dir']):
#             df = pd.read_csv('%s/relabeled_data.csv' % args['data_dir'])
#         else:
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
            self.dgraph_path = '../data/saved_graphs/full_%s_dgraph_sep.bin' % args['dataset']
        else:
            self.labels = [eval(t) for t in df['Labels_mix']]
            self.dgraph_path = '../data/saved_graphs/full_%s_dgraph_mix.bin' % args['dataset']
        self._pre_process(mol_to_graph, node_featurizer, edge_featurizer, load, log_every)
    
    def _pre_process(self, mol_to_graph, node_featurizer, edge_featurizer):
        self.fgraphs_exist = os.path.exists(self.fgraph_path)
        self.dgraphs_exist = os.path.exists(self.dgraph_path)
        self.fgraphs = []
        self.dgraphs = []
                    
        if not self.fgraphs_exist or not self.dgraphs_exist:   
            for (s1, s2) in tqdm(zip(self.reactants, self.reagents), total=len(self.reactants), desc='Building dgl graphs...'):
                smiles = combine_reactants(s1, s2)
                if not self.fgraphs_exist:
                    mol = Chem.MolFromSmiles(smiles)
                    self.fgraphs.append(mol_to_graph(mol, node_featurizer=node_featurizer, edge_featurizer=edge_featurizer, canonical_atom_order=False))
                if  not self.dgraphs_exist:
                    if self.sep:
                        self.dgraphs.append(make_densegraph(s1, mol_to_graph))
                    else:
                        self.dgraphs.append(make_densegraph(smiles, mol_to_graph))
                        
        if self.fgraphs_exist:
            print ('Loading feture graphs from %s...' % self.fgraph_path)
            self.fgraphs, _ = load_graphs(self.fgraph_path)
        else:
            save_graphs(self.fgraph_path, self.fgraphs)

        if self.dgraphs_exist:
            print ('Loading dense graphs from %s...' % self.dgraph_path)
            self.dgraphs, _ = load_graphs(self.dgraph_path)
        else:
            save_graphs(self.dgraph_path, self.dgraphs)
        [g.ndata.pop('_ID') for g in self.dgraphs if '_ID' in g.ndata]
        [g.edata.pop('_ID') for g in self.dgraphs if '_ID' in g.edata]
            
    def __getitem__(self, item):
        return self.reactants[item], self.reagents[item], self.fgraphs[item], self.dgraphs[item], self.labels[item], self.masks[item]

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
            self.dgraph_path = '../data/saved_graphs/test_%s_dgraph_sep.bin' % args['dataset']
        else:
            self.dgraph_path = '../data/saved_graphs/test_%s_dgraph_mix.bin' % args['dataset']
        self._pre_process(mol_to_graph, node_featurizer, edge_featurizer)
        
    
    def _pre_process(self, mol_to_graph, node_featurizer, edge_featurizer):
        self.fgraphs_exist = os.path.exists(self.fgraph_path)
        self.dgraphs_exist = os.path.exists(self.dgraph_path)
        self.fgraphs = []
        self.dgraphs = []
                    
        if not self.fgraphs_exist or not self.dgraphs_exist:   
            for (s1, s2) in tqdm(zip(self.reactants, self.reagents), total=len(self.reactants), desc='Building dgl graphs...'):
                smiles = combine_reactants(s1, s2)
                if not self.fgraphs_exist:
                    mol = Chem.MolFromSmiles(smiles)
                    self.fgraphs.append(mol_to_graph(mol, node_featurizer=node_featurizer, edge_featurizer=edge_featurizer, canonical_atom_order=False))
                if  not self.dgraphs_exist:
                    if self.sep:
                        self.dgraphs.append(make_densegraph(s1, mol_to_graph))
                    else:
                        self.dgraphs.append(make_densegraph(smiles, mol_to_graph))
                        
        if self.fgraphs_exist:
            print ('Loading feture graphs from %s...' % self.fgraph_path)
            self.fgraphs, _ = load_graphs(self.fgraph_path)
        else:
            save_graphs(self.fgraph_path, self.fgraphs)

        if self.dgraphs_exist:
            print ('Loading dense graphs from %s...' % self.dgraph_path)
            self.dgraphs, _ = load_graphs(self.dgraph_path)
        else:
            save_graphs(self.dgraph_path, self.dgraphs)
            
    def __getitem__(self, item):
            return self.reactants[item], self.reagents[item], self.fgraphs[item], self.dgraphs[item]

    def __len__(self):
            return len(self.reactants)
        