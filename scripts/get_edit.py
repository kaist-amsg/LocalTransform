import os
import numpy as np
import pandas as pd

import time
import torch
import torch.nn as nn
import dgl

from utils import predict

def get_id_template(a, class_n):
    edit_idx = a//class_n
    template = a%class_n
    return (edit_idx, template)

def logit2pred(out, top_num):
    class_n = out.size(-1)
    readout = out.cpu().detach().numpy()
    readout = readout.reshape(-1)
    output_rank = np.flip(np.argsort(readout))
    output_rank = [r for r in output_rank if get_id_template(r, class_n)[1] != 0][:top_num]
    selected_site = [get_id_template(a, class_n) for a in output_rank]
    selected_proba = [readout[a] for a in output_rank]
     
    return selected_site, selected_proba
    
def get_bond_site(graph, etype):
    atom_pair_list = torch.transpose(graph.adjacency_matrix(etype=etype).coalesce().indices(), 0, 1).numpy()
    atom_pair_list = [atom_pair_list_v[idx] for idx in pred_vi]
    
def combined_edit(virtual_bonds, real_bonds, pred_vi, pred_ri, pred_v, pred_r, top_num):
    pred_site_v, pred_score_v = logit2pred(pred_v, top_num)
    pred_site_r, pred_score_r = logit2pred(pred_r, top_num)
    pooled_virtual_bonds = [virtual_bonds[idx] for idx in pred_vi]
    pooled_real_bonds = [real_bonds[idx] for idx in pred_ri]
    pred_site_v = [(list(pooled_virtual_bonds[pred_site]), pred_temp)  for pred_site, pred_temp in pred_site_v]
    pred_site_r = [(list(pooled_real_bonds[pred_site]), pred_temp)  for pred_site, pred_temp in pred_site_r]
    
    pred_sites = pred_site_v + pred_site_r
    pred_types = ['v'] * top_num + ['r'] * top_num
    pred_scores = pred_score_v + pred_score_r
    pred_ranks = np.flip(np.argsort(pred_scores))[:top_num]
    
    pred_types = [pred_types[r] for r in pred_ranks]
    pred_sites = [pred_sites[r] for r in pred_ranks]
    pred_scores = [pred_scores[r] for r in pred_ranks]
    return pred_types, pred_sites, pred_scores

def get_bg_partition(bg, bonds_dicts):
    gs = dgl.unbatch(bg)
    v_sep = [0]
    r_sep = [0]
    for g, v_bonds, r_bonds in zip(gs, bonds_dicts['virtual'], bonds_dicts['real']):
        pooling_size = g.num_nodes()
        n_v_bonds = len(v_bonds)
        if n_v_bonds < pooling_size:
            v_sep.append(v_sep[-1] + n_v_bonds)
        else:
            v_sep.append(v_sep[-1] + pooling_size)
        n_r_bonds = len(r_bonds)
        if n_r_bonds < pooling_size:    
            r_sep.append(r_sep[-1] + n_r_bonds)
        else:
            r_sep.append(r_sep[-1] + pooling_size)
    return v_sep[1:], r_sep[1:]

def write_edits(args, model, test_loader):
    model.eval()
    with open(args['result_path'], 'w') as f:
        f.write('Test_id\tReactants\tReagents\t%s\n' % '\t'.join(['Prediction %s' % (i+1) for i in range(args['top_num'])]))
        with torch.no_grad():
            for batch_id, data in enumerate(test_loader):
                reactants, reagents, bg, adms, bonds_dicts = data
                pred_VT, pred_RT, _, _, pred_VI, pred_RI, _  = predict(args, model, bg, adms, bonds_dicts)
                pred_VT = nn.Softmax(dim=1)(pred_VT)
                pred_RT = nn.Softmax(dim=1)(pred_RT)
                v_sep, r_sep = get_bg_partition(bg, bonds_dicts)
                start_v = 0
                start_r = 0
                print('\rWriting test molecule batch %s/%s' % (batch_id, len(test_loader)), end='', flush=True)
                for i, (reactant, reagent) in enumerate(zip(reactants, reagents)):
                    end_v, end_r = v_sep[i], r_sep[i]
                    virtual_bonds, real_bonds = bonds_dicts['virtual'][i].numpy(), bonds_dicts['real'][i].numpy()
                    pred_vi, pred_ri = pred_VI[i].cpu(), pred_RI[i].cpu()
                    pred_v, pred_r = pred_VT[start_v:end_v], pred_RT[start_r:end_r]
                    
                    pred_types, pred_sites, pred_scores = combined_edit(virtual_bonds, real_bonds, pred_vi, pred_ri, pred_v, pred_r, args['top_num'])
                    test_id = (batch_id * args['batch_size']) + i
                    f.write('%s\t%s\t%s\t%s\n' % (test_id, reactant, reagent, '\t'.join(['(%s, %s, %s, %.3f)' % (pred_types[i], pred_sites[i][0], pred_sites[i][1], pred_scores[i]) for i in range(args['top_num'])])))
                    start_v = end_v
                    start_r = end_r
    print ()
    return 
