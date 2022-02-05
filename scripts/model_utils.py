# code of Multi-head Attention is modified from https://github.com/SamLynnEvans/Transformer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import math
import copy
from collections import defaultdict
from itertools import permutations

import sklearn
import dgl
import dgllife

def pair_atom_feats(g, node_feats, etype):
    atom_pair_list = torch.transpose(g.adjacency_matrix(etype=etype).coalesce().indices(), 0, 1)
    atom_pair_idx1 = atom_pair_list[:,0]
    atom_pair_idx2 = atom_pair_list[:,1]
    atom_pair_feats = torch.cat((node_feats[atom_pair_idx1], node_feats[atom_pair_idx2]), dim = 1)
    return atom_pair_feats

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
    gs = dgl.unbatch(bg)
    atom_feats = [feats[:g.num_nodes()] for feats, g in zip(atom_feats, gs)]
    bg.ndata['h'] = torch.cat(atom_feats, dim = 0)
    rgs = dgl.unbatch(bg)
    hgs = dgl.unbatch(hg)
    for rg, hg in zip(rgs, hgs):
        reactant_feats.append(rg.ndata['h'][:hg.num_nodes()])
    return torch.cat(reactant_feats, dim = 0)
   
def reactive_pooling(hg, atom_feats, PoolingNet, etype):
    device = atom_feats.device
    react_outputs = []
    pool_feats = []
    top_idxs = []
    hg.ndata['h'] = atom_feats
    gs = dgl.unbatch(hg)
    for g in gs:
        n_edges = g.num_edges(etype=etype)
        if n_edges == 0:
            pool_feats.append(torch.FloatTensor([]).to(device))
            top_idxs.append(torch.LongTensor([]).to(device))
            continue
        bond_feats = pair_atom_feats(g, g.ndata['h'], etype)
        react_output = PoolingNet(bond_feats)
        pooling_size = g.num_nodes()
        if n_edges < pooling_size:
            _, top_idx = torch.topk(react_output[:,1], n_edges)
        else:
            _, top_idx = torch.topk(react_output[:,1], pooling_size)
        react_outputs.append(react_output[top_idx])
        pool_feats.append(bond_feats[top_idx])
        top_idxs.append(top_idx)
        
    react_outputs = torch.cat(react_outputs, dim = 0)
    return react_outputs, pool_feats, top_idxs

def pack_bond_feats(feats_v, feats_r):
    edit_feats = [torch.cat([v, r], dim = 0) for v, r in zip(feats_v, feats_r)]
    masks = [torch.ones(len(feats), dtype=torch.uint8) for feats in edit_feats]
    padded_feats = pad_sequence(edit_feats, batch_first=True, padding_value= 0)
    masks = pad_sequence(masks, batch_first=True, padding_value= 0)
    return padded_feats, masks

def unpack_bond_feats(edit_feats, vritual_idx, real_idx):
    feats_v = []
    feats_r = []
    for feats, vi, ri in zip(edit_feats, vritual_idx, real_idx):
        vn, rn = len(vi), len(ri)
        feats_v.append(feats[:vn])
        feats_r.append(feats[vn:vn+rn])
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
            self.relative_v = nn.Parameter(torch.randn(self.p_k, self.d_k))
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
        
    def one_hot_embedding(self, labels):
        y = torch.eye(self.p_k)
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
            gpms = self.one_hot_embedding(gpm.unsqueeze(1).repeat(1, self.h, 1, 1)).to(x.device)
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
        return x + self.dropout2(output), attn
        
    
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
#         x = self.layer_norm(x)
#         output = self.net(x)
        output = self.net(self.layer_norm(x))
        return x + self.dropout(output)
    
class Global_Reactivity_Attention(nn.Module):
    def __init__(self, d_model, heads, n_layers = 6, positional_number = 5, dropout = 0.1):
        super(Global_Reactivity_Attention, self).__init__()
        self.n_layers = n_layers
        att_stack = []
        pff_stack = []
        for _ in range(n_layers-1):
            att_stack.append(MultiHeadAttention(heads, d_model, positional_number, dropout))
            pff_stack.append(FeedForward(d_model, dropout=dropout))
        if n_layers > 1:
            self.att_stack = nn.ModuleList(att_stack)
            self.pff_stack = nn.ModuleList(pff_stack)
        self.last_att = MultiHeadAttention(heads, d_model, positional_number, dropout)
        
    def forward(self, x, rpm, mask = None):
        scores = []
        for n in range(self.n_layers-1):
            x, score = self.att_stack[n](x, rpm, mask)
            scores.append(score)
            x = self.pff_stack[n](x)
            
        output, score = self.last_att(x, rpm, mask)
        scores.append(score)
        return output, scores