import torch
import torch.nn as nn

import sklearn
import dgl
import dgllife
from dgllife.model import MPNNGNN
from model_utils import pack_atom_feats, unpack_atom_feats, pack_bond_feats, unpack_bond_feats, reactive_pooling, Global_Reactivity_Attention

class LocalTransform(nn.Module):
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
        super(LocalTransform, self).__init__()
        
        self.activation = nn.ReLU()
        
        self.mpnn = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)
        
        self.atom_att = Global_Reactivity_Attention(node_out_feats, attention_heads, attention_layers, 8)
        self.bond_att = Global_Reactivity_Attention(node_out_feats*2, attention_heads, attention_layers, 4)
        
        self.poolings =  {'virtual': nn.Sequential(
                                    nn.Linear(node_out_feats*2, node_out_feats),
                                    self.activation, 
                                    nn.Dropout(0.2),
                                    nn.Linear(node_out_feats, 2)), 
                          'real': nn.Sequential(
                                    nn.Linear(node_out_feats*2, node_out_feats),
                                    self.activation, 
                                    nn.Dropout(0.2),
                                    nn.Linear(node_out_feats, 2))}
        
        self.output_v =  nn.Sequential(
                            nn.Linear(node_out_feats*2, node_out_feats), 
                            self.activation, 
                            nn.Dropout(0.2),
                            nn.Linear(node_out_feats, Template_vn+1))
        
        self.output_r =  nn.Sequential(
                            nn.Linear(node_out_feats*2, node_out_feats), 
                            self.activation, 
                            nn.Dropout(0.2),
                            nn.Linear(node_out_feats, Template_rn+1))
                
    def forward(self, bg, adms, bonds_dict, node_feats, edge_feats):
        
        atom_feats = self.mpnn(bg, node_feats, edge_feats)
        atom_feats, mask = pack_atom_feats(bg, atom_feats)
        atom_feats, atom_attention = self.atom_att(atom_feats, adms, mask)
        idxs_dict, rout_dict, bonds_feats, bonds = reactive_pooling(bg, atom_feats, bonds_dict, self.poolings)
        bond_feats, mask, bcms = pack_bond_feats(bonds_feats, bonds)
        bond_feats, bond_attention = self.bond_att(bond_feats, bcms, mask)
        feats_v, feats_r = unpack_bond_feats(bond_feats, idxs_dict)
        template_v, template_r = self.output_v(feats_v), self.output_r(feats_r)
        
        return template_v, template_r, rout_dict['virtual'], rout_dict['real'], idxs_dict['virtual'], idxs_dict['real'], (atom_attention, bond_attention)
