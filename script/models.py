import torch
import torch.nn as nn

import sklearn
import dgl
import dgllife
from dgllife.model import MPNNGNN
from model_utils import get_reactant_feats, reactive_group_pooling, Global_Reactivity_Attention, unpack_feats, pack_feats, pack_bond_feats, unpack_bond_feats, get_bondmatrix

class NeuralChemist(nn.Module):
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats,
                 edge_hidden_feats,
                 num_step_message_passing,
                 attention_heads,
                 attention_layers,
                 Template_rn, 
                 Template_vn):
        super(NeuralChemist, self).__init__()
        
        self.activation = nn.ReLU()
        
        self.mpnn = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)
        
        self.AttentionNet_atom = Global_Reactivity_Attention(node_out_feats, attention_heads, attention_layers, 8)
        
        self.AttentionNet_bond = Global_Reactivity_Attention(node_out_feats*2, attention_heads, attention_layers, 0)
        
        
        self.PoolingNet_v = nn.Sequential(
                            nn.Linear(node_out_feats*2, node_out_feats),
                            self.activation, 
                            nn.Dropout(0.2),
                            nn.Linear(node_out_feats, 2))
 
        self.PoolingNet_r =  nn.Sequential(
                            nn.Linear(node_out_feats*2, node_out_feats), 
                            self.activation, 
                            nn.Dropout(0.2),
                            nn.Linear(node_out_feats, 2))
        
        self.OutputNet_v =  nn.Sequential(
                            nn.Linear(node_out_feats*2, node_out_feats), 
                            self.activation, 
                            nn.Dropout(0.2),
                            nn.Linear(node_out_feats, Template_vn+1))
        
        self.OutputNet_r =  nn.Sequential(
                            nn.Linear(node_out_feats*2, node_out_feats), 
                            self.activation, 
                            nn.Dropout(0.2),
                            nn.Linear(node_out_feats, Template_rn+1))
        
                
    def forward(self, apms, bg, hg, node_feats, edge_feats):
        atom_feats = self.mpnn(bg, node_feats, edge_feats)
        atom_feats, mask = pack_feats(bg, atom_feats)
        atom_feats, attention_score = self.AttentionNet_atom(atom_feats, apms, mask)
        atom_feats = unpack_feats(bg, atom_feats)
        atom_feats = get_reactant_feats(bg, hg, atom_feats)
        react_v, feats_v, idxs_v = reactive_group_pooling(hg, atom_feats, self.PoolingNet_v, 'virtual')
        react_r, feats_r, idxs_r = reactive_group_pooling(hg, atom_feats, self.PoolingNet_r, 'real')
        
        bond_feats, mask = pack_bond_feats(feats_v, feats_r)

        bond_feats, _ = self.AttentionNet_bond(bond_feats, None, mask)
        feats_v, feats_r = unpack_bond_feats(bond_feats, idxs_v, idxs_r)
        
        template_v, template_r = self.OutputNet_v(feats_v), self.OutputNet_r(feats_r)
        
        return template_v, template_r, react_v, react_r, idxs_v, idxs_r, attention_score
