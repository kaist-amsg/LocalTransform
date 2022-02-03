import torch
import sklearn
import dgl
import errno
import json
import os
import time
import numpy as np
import pandas as pd
from functools import partial

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler

from rdkit import Chem
from dgl.data.utils import Subset
from dgllife.utils import WeaveAtomFeaturizer, CanonicalBondFeaturizer, mol_to_bigraph, EarlyStopping

from models import NeuralChemist
from dataset import USPTODataset, USPTOTestDataset, combine_reactants

def init_featurizer(args):
    atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
             'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
             'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
             'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir',
             'Ce', 'Gd', 'Ga', 'Cs']
    args['node_featurizer'] = WeaveAtomFeaturizer(atom_types = atom_types)
    args['edge_featurizer'] = CanonicalBondFeaturizer(self_loop=True)
    return args

def get_configure(args):
    with open('%s.json' % args['config_path'], 'r') as f:
        config = json.load(f)
    config['Template_rn'] = len(pd.read_csv('%s/real_templates.csv' % args['data_dir']))
    config['Template_vn'] = len(pd.read_csv('%s/virtual_templates.csv' % args['data_dir']))
    args['Template_rn'] = config['Template_rn']
    args['Template_vn'] = config['Template_vn']
    config['in_node_feats'] = args['node_featurizer'].feat_size()
    config['in_edge_feats'] = args['edge_featurizer'].feat_size()
    return config

def mkdir_p(path):
    try:
        os.makedirs(path)
        print('Created directory %s'% path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print('Directory %s already exists.' % path)
        else:
            raise

def load_dataloader(args):
    if args['mode'] == 'train':
        dataset = USPTODataset(args, 
                            mol_to_graph=partial(mol_to_bigraph, add_self_loop=True),
                            node_featurizer=args['node_featurizer'],
                            edge_featurizer=args['edge_featurizer'])

        train_set, val_set, test_set = Subset(dataset, dataset.train_ids), Subset(dataset, dataset.val_ids), Subset(dataset, dataset.test_ids)

        train_loader = DataLoader(dataset=train_set, batch_size=args['batch_size'], shuffle=True,
                                  collate_fn=collate_molgraphs, num_workers=args['num_workers'])
        val_loader = DataLoader(dataset=val_set, batch_size=args['batch_size'],
                                collate_fn=collate_molgraphs, num_workers=args['num_workers'])
        test_loader = DataLoader(dataset=test_set, batch_size=args['batch_size'],
                                 collate_fn=collate_molgraphs, num_workers=args['num_workers'])
        return train_loader, val_loader, test_loader
    
    elif args['mode'] == 'test':
        test_set = USPTOTestDataset(args, 
                            mol_to_graph=partial(mol_to_bigraph, add_self_loop=True),
                            node_featurizer=args['node_featurizer'],
                            edge_featurizer=args['edge_featurizer'])
        test_loader = DataLoader(dataset=test_set, batch_size=args['batch_size'],
                                 collate_fn=collate_molgraphs_test, num_workers=args['num_workers'])
        
    elif args['mode'] == 'analyze':
        test_set = USPTOAnaynizeDataset(args, 
                            mol_to_graph=partial(mol_to_bigraph, add_self_loop=True),
                            node_featurizer=args['node_featurizer'],
                            edge_featurizer=args['edge_featurizer'])
        test_loader = DataLoader(dataset=test_set, batch_size=args['batch_size'],
                                 collate_fn=collate_molgraphs_analysis, num_workers=args['num_workers'])
    return test_loader

def weight_loss(class_num, args, reduction = 'none'):
    weights=torch.ones(class_num+1)
    weights[0] = args['negative_weight']
    return nn.CrossEntropyLoss(weight = weights.to(args['device']), reduction = reduction)

def load_model(args):
    exp_config = get_configure(args)
    model = NeuralChemist(
        node_in_feats=exp_config['in_node_feats'],
        edge_in_feats=exp_config['in_edge_feats'],
        node_out_feats=exp_config['node_out_feats'],
        edge_hidden_feats=exp_config['edge_hidden_feats'],
        num_step_message_passing=exp_config['num_step_message_passing'],
        attention_heads = exp_config['attention_heads'],
        attention_layers = exp_config['attention_layers'],
        Template_rn = exp_config['Template_rn'],
        Template_vn = exp_config['Template_vn'])
    model = model.to(args['device'])
    print ('Parameters of loaded LocalTransform:')
    print (exp_config)

    if args['mode'] == 'train':
        loss_criterions = [weight_loss(1, args), weight_loss(exp_config['Template_vn'], args), weight_loss(exp_config['Template_rn'], args)]
        
        optimizer = Adam(model.parameters(), lr = args['learning_rate'], weight_decay = args['weight_decay'])
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args['schedule_step'])
#         scheduler = lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.8)
        if os.path.exists(args['model_path']):
            user_answer = input('%s exists, want to (a) overlap (b) continue from checkpoint (c) make a new model?' % args['model_path'])
            if user_answer == 'a':
                stopper = EarlyStopping(mode = 'lower', patience=args['patience'], filename=args['model_path'])
                print ('Overlap exsited model and training a new model...')
            elif user_answer == 'b':
                stopper = EarlyStopping(mode = 'lower', patience=args['patience'], filename=args['model_path'])
                stopper.load_checkpoint(model)
                print ('Train from existed model checkpoint...')
            elif user_answer == 'c':
                model_name = input('Enter new model name: ')
                args['model_path'] = '../models/%s.pth' % model_name
                stopper = EarlyStopping(mode = 'lower', patience=args['patience'], filename=args['model_path'])
                print ('Training a new model %s.pth' % model_name)
        else:
            stopper = EarlyStopping(mode = 'lower', patience=args['patience'], filename=args['model_path'])
        return model, loss_criterions, optimizer, scheduler, stopper
    
    else:
        model.load_state_dict(torch.load(args['model_path'])['model_state_dict'])
        return model

def make_labels(hgraphs, labels, masks):
    vtemplate_labels = []
    rtemplate_labels = []
    for hg, label, m in zip(hgraphs, labels, masks):
        edge_r = hg.number_of_edges(etype='real')
        edge_v = hg.number_of_edges(etype='virtual')
        vtemplate_label, rtemplate_label = [0]*(edge_v), [0]*(edge_r)
        if m == 1:
            for l in label:
                edit_type = l[0]
                edit_idx = l[1]
                edit_template = l[2]
                if edit_type == 'v':
                    vtemplate_label[edit_idx] = edit_template
                else:
                    rtemplate_label[edit_idx] = edit_template

        vtemplate_labels.append(vtemplate_label)
        rtemplate_labels.append(rtemplate_label)    
    return vtemplate_labels, rtemplate_labels

def get_reactive_template_labels(all_template_labels, masks, top_idxs):
    template_labels = []
    clipped_masks = []
    for i, idxs in enumerate(top_idxs):
        template_labels += [all_template_labels[i][idx] for idx in idxs]
        clipped_masks +=  [masks[i] for idx in idxs]
    reactive_labels = [int(y > 0) for y in template_labels]
    return torch.LongTensor(template_labels), torch.LongTensor(reactive_labels), torch.LongTensor(clipped_masks)

def atom_position_matrix(reactants, reagents, max_distance = 6):
    dms = []
    smiles = [combine_reactants(r1, r2) for r1, r2 in zip(reactants, reagents)]
    max_size = max([Chem.MolFromSmiles(s).GetNumAtoms() for s in smiles])
    for s in smiles:
        temp = np.ones((max_size, max_size)) * max_distance + 1
        m = Chem.MolFromSmiles(s)
        dm = Chem.GetDistanceMatrix(m)
        dm[dm > 100] = -1
        dm[dm > max_distance] = max_distance
        dm[dm == -1] = max_distance + 1
        temp[:dm.shape[0],:dm.shape[1]] = dm
        dms.append(torch.tensor(temp).unsqueeze(0).long())
    return torch.cat(dms, dim = 0)

def collate_molgraphs(data):
    reactants, reagents, graphs, hgraphs, labels, masks = map(list, zip(*data))
    true_VT, true_RT = make_labels(hgraphs, labels, masks)
    hbg = dgl.batch(hgraphs)
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    dms = atom_position_matrix(reactants, reagents)
    return reactants, dms, bg, hbg, true_VT, true_RT, masks

def collate_molgraphs_test(data):
    reactants, reagents, graphs, hgraphs = map(list, zip(*data))
    hbg = dgl.batch(hgraphs)
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    dms = atom_position_matrix(reactants, reagents)
    return reactants, reagents, dms, bg, hbg


def predict(args, model, rpms, bg, hbg):
    rpms = rpms.to(args['device'])
    bg = bg.to(args['device'])
    hbg = hbg.to(args['device'])
    node_feats = bg.ndata.pop('h').to(args['device'])
    edge_feats = bg.edata.pop('e').to(args['device'])
    return model(rpms, bg, hbg, node_feats, edge_feats)
