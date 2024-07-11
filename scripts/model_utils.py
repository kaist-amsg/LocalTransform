# code of Multi-head Attention is modified from https://github.com/SamLynnEvans/Transformer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import math
import copy
import numpy as np
from collections import defaultdict
from itertools import permutations

import sklearn
import dgl
import dgllife

from time import time

def pack_atom_feats(bg, atom_feats):
    bg.ndata['h'] = atom_feats
    gs = dgl.unbatch(bg)
    edit_feats = [g.ndata['h'] for g in gs]
    masks = [torch.ones(g.num_nodes(), dtype=torch.uint8) for g in gs]
    padded_feats = pad_sequence(edit_feats, batch_first=True, padding_value= 0)
    masks = pad_sequence(masks, batch_first=True, padding_value= 0)
    
    return padded_feats, masks

def unpack_atom_feats(bg, hg, atom_feats):
    reactant_feats = []
    atom_feats = [feats[:g.num_nodes()] for feats, g in zip(atom_feats, dgl.unbatch(bg))]
    bg.ndata['h'] = torch.cat(atom_feats, dim = 0)
    rgs = dgl.unbatch(bg)
    hgs = dgl.unbatch(hg)
    for rg, hg in zip(rgs, hgs):
        reactant_feats.append(rg.ndata['h'][:hg.num_nodes()])
    return torch.cat(reactant_feats, dim = 0)

def reactive_pooling(bg, atom_feats, bonds_dict, pooling_nets, bond_nets):
    feat_dim, device = atom_feats.size(-1), atom_feats.device
    react_outputs = {'virtual':[], 'real':[]}
    top_idxs = {'virtual':[], 'real':[]}
    pooled_feats_batch = []
    pooled_bonds_batch = []
    bg.ndata['h'] = torch.cat([feats[:g.num_nodes()] for feats, g in zip(atom_feats, dgl.unbatch(bg))], dim = 0)
    gs = dgl.unbatch(bg)
    empty_longtensor = torch.LongTensor([]).to(device)
    empty_floattensor = torch.FloatTensor([]).to(device)
    for i, g in enumerate(gs):
        pool_size = g.num_nodes()
        pooled_feats = []
        pooled_bonds = []
        for bond_type in ['virtual', 'real']:
            bonds = bonds_dict[bond_type][i]
            pooling_net, bond_net = pooling_nets[bond_type], bond_nets[bond_type]
            n_bonds = bonds.size(0)
            if n_bonds == 0:
                top_idxs[bond_type].append(empty_longtensor)
                react_outputs[bond_type].append(empty_floattensor)
                continue

            bond_feats = g.ndata['h'][bonds.unsqueeze(1)].view(-1, feat_dim*2)
            react_output = pooling_net(bond_feats)
            if n_bonds < pool_size:
                _, top_idx = torch.topk(react_output[:,1], n_bonds)
            else:
                _, top_idx = torch.topk(react_output[:,1], pool_size)

            top_idxs[bond_type].append(top_idx)
            react_outputs[bond_type].append(react_output[top_idx])
            pooled_feats.append(bond_net(bond_feats[top_idx]))
            pooled_bonds.append(bonds[top_idx.cpu()])
            
        pooled_feats_batch.append(torch.cat(pooled_feats))
        pooled_bonds_batch.append(torch.cat(pooled_bonds))
        
    for bond_type in ['virtual', 'real']:
        react_outputs[bond_type] = torch.cat(react_outputs[bond_type], dim = 0)
    return top_idxs, react_outputs, pooled_feats_batch, pooled_bonds_batch

def get_bdm(bonds, max_size):  # bond distance matrix
    temp = torch.eye(max_size)
    for i, bond1 in enumerate(bonds):
        for j, bond2 in enumerate(bonds):
            if i >= j: 
                continue
            if torch.unique(torch.cat([bond1, bond2])).size(0) < 4:  # at least on overlap
                temp[i][j], temp[j][i] = 1, 1 # connect
    return temp.unsqueeze(0).long() 

def pack_bond_feats(bonds_feats, pooled_bonds):
    masks = [torch.ones(len(feats), dtype=torch.uint8) for feats in bonds_feats]
    padded_feats = pad_sequence(bonds_feats, batch_first=True, padding_value= 0)
    bdms = [get_bdm(bonds, padded_feats.size(1)) for bonds in pooled_bonds]
    masks = pad_sequence(masks, batch_first=True, padding_value= 0)
    return padded_feats, masks, torch.cat(bdms, dim = 0)

def unpack_bond_feats(bond_feats, idxs_dict):
    feats_v = []
    feats_r = []
    for feats, v_bonds, r_bonds in zip(bond_feats, idxs_dict['virtual'], idxs_dict['real']):
        n_vbonds, n_rbonds = v_bonds.size(0), r_bonds.size(0)
        feats_v.append(feats[:n_vbonds])
        feats_r.append(feats[n_vbonds:n_vbonds+n_rbonds])
    return torch.cat(feats_v, dim = 0), torch.cat(feats_r, dim = 0)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, positional_number = 5, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.p_k = positional_number
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        if self.p_k != 0:
            self.relative_k = nn.Parameter(torch.randn(self.p_k, self.d_k))
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Sequential(
                            nn.Linear(d_model, d_model), 
                            nn.ReLU(), 
                            nn.Dropout(dropout),
                            nn.Linear(d_model, d_model))
        self.gating = nn.Linear(d_model, d_model)
        self.to_out = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.reset_parameters()
        
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.constant_(self.gating.weight, 0.)
        nn.init.constant_(self.gating.bias, 1.)
        
    def one_hot_embedding(self, labels, device):
        y = torch.eye(self.p_k).to(device)
        return y[labels]
                
    def forward(self, x, gpm, mask=None):
        bs, atom_size = x.size(0), x.size(1)
        x = self.layer_norm(x)
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        k1 = k.view(bs, -1, self.h, self.d_k).transpose(1,2)
        q1 = q.view(bs, -1, self.h, self.d_k).transpose(1,2)
        v1 = v.view(bs, -1, self.h, self.d_k).transpose(1,2)
        attn1 = torch.matmul(q1, k1.permute(0, 1, 3, 2))
        
        if self.p_k == 0:
            attn = attn1/math.sqrt(self.d_k)
        else:
            gpms = self.one_hot_embedding(gpm.unsqueeze(1).repeat(1, self.h, 1, 1), x.device)
            attn2 = torch.matmul(q1, self.relative_k.transpose(0, 1))
            attn2 = torch.matmul(gpms, attn2.unsqueeze(-1)).squeeze(-1)
            attn = (attn1 + attn2) /math.sqrt(self.d_k)
        
        if mask is not None:
            mask = mask.bool()
            mask = mask.unsqueeze(1).repeat(1,mask.size(-1),1)
            mask = mask.unsqueeze(1).repeat(1,attn.size(1),1,1)
            attn[~mask] = float(-9e9)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout1(attn)
        v1 = v.view(bs, -1, self.h, self.d_k).permute(0, 2, 1, 3)
        output = torch.matmul(attn, v1)

        output = output.transpose(1,2).contiguous().view(bs, -1, self.d_model).squeeze(-1)
        output = self.to_out(output * self.gating(x).sigmoid()) # gate self attention
        return self.dropout2(output), attn
#         return output, attn
        
    
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * torch.pow(x, 3)))) 
    
class FeedForward(nn.Module):
    def __init__(self, d_model, dropout = 0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*2, d_model))
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.layer_norm(x)
        return self.net(x)
    
class Global_Reactivity_Attention(nn.Module):
    def __init__(self, d_model, heads = 8, n_layers = 3, positional_number = 5, dropout = 0.1):
        super(Global_Reactivity_Attention, self).__init__()
        self.n_layers = n_layers
        att_stack = []
        pff_stack = []
        for _ in range(n_layers):
            att_stack.append(MultiHeadAttention(heads, d_model, positional_number, dropout))
            pff_stack.append(FeedForward(d_model, dropout=dropout))
        self.att_stack = nn.ModuleList(att_stack)
        self.pff_stack = nn.ModuleList(pff_stack)
        
    def forward(self, x, rpm, mask = None):
        att_scores = {}
        for n in range(self.n_layers):
            m, att_score = self.att_stack[n](x, rpm, mask)
            x = x + self.pff_stack[n](x+m)
            att_scores[n] = att_score
        return x, att_scores