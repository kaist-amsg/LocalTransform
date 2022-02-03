import copy
import numpy as np
from functools import reduce
from collections import defaultdict
from itertools import permutations

import rdkit
from rdkit import Chem, RDLogger

from .template_decoder import *

def demap(smiles):
    if smiles == None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]     
    return Chem.MolToSmiles(mol)

def combine_dict(d1, d2):
    for k, v in d1.items():
        if k in d2:
            if v != d2[k]:
                return False
        elif v in d2.values():
            return False
        else:
            d2[k] = v
    return d2
    
def match_each(preds, trues, matched_idx, ):
    perms = list(permutations(preds, len(trues)))
    ts = [item for elem in trues for item in elem]
    ms = []
    for perm in perms:
        ps = [item for elem in perm for item in elem]
        m = {t:p for t, p in zip(ts, ps) if t != -1}
        combined_dict = combine_dict(m, copy.copy(matched_idx))
        if combined_dict:
            ms.append(combined_dict)
    return ms

def bidirect_len(bonds, return_len = True):
    bidirected = copy.copy(bonds)
    for bond in bonds:
        if (bond[1], bond[0]) not in bidirected:
            bidirected.append((bond[1], bond[0]))
    if return_len:
        return len(bidirected)
    else:
        return bidirected

def single_match(pred_action, pred_idx, template_actions):
    template_idxs = template_actions[pred_action]
    matches = []
    for temp_idx in template_idxs:
        matches.append({t:p for t, p in zip(temp_idx, pred_idx) if t != -1})
    return matches

def split_pred_idxs(pred_idxs):
    matched_idxs = []
    for vs in pred_idxs:
        matched_idxs += [v for v in vs if v not in matched_idxs]
    return matched_idxs
    
class Collector():
    def __init__(self, reactant, tempaltes_info, reagents, products = 'nan', intermolecule = False, verbose = False):
        self.templates_info = tempaltes_info
        self.reactant = reactant
        if str(reagents) == 'nan':
            self.reagents = ''
        else:
            self.reagents = reagents
        
        if products != None:
            self.min_n_atoms = min([8] + [Chem.MolFromSmiles(p).GetNumAtoms() for p in products.split('.') if Chem.MolFromSmiles(p).GetNumAtoms() != 1])
            d_product = '.'.join([p for p in demap(products).split('.') if p not in demap(self.reactant).split('.') + demap(self.reagents).split('.')])
            self.products = self.clean_small_frags(d_product)
            if self.products == d_product:
                self.has_small_fragment = False
            else:
                self.has_small_fragment = True
            self.non_reacts = self.get_nonreact_frags()
        else:
            self.min_n_atoms = 1
            self.products = None
            self.non_reacts = []
            self.has_small_fragment = False
            
        self.verbose = verbose
        if self.verbose:
            print ('product:', self.products)
        self.intermolecule = intermolecule
        
        self.predictions = defaultdict(dict)
        self.old_predictions = set()
        self.predicted_template = defaultdict(list)
        self.template_scores = defaultdict(dict)
        self.used_idx = defaultdict(list)
        self.predicted_roles = dict()
        
        
    def clean_small_frags(self, products):
        return  '.'.join([product for product in products.split('.') if Chem.MolFromSmiles(product).GetNumAtoms() >= self.min_n_atoms])

    def fix_small_product(self, products):
        if '[IH3]' in products:
            return products.replace('[IH3]', '[IH]')
        return self.clean_small_frags(products)

    def get_nonreact_frags(self):
        non_reacts = []
        for frag in self.reactant.split('.'):
            if frag in self.products.split('.'):
                non_reacts.append(frag)
        return non_reacts
        
    def structure_match(self, pred, true):
        if true == None:
            return False
        elif true == '':
            return True
        try:
            trues = [true for true in demap(true).split('.')]
            pred_mix = [r for r in demap(self.reactant).split('.')] + pred.split('.')
            for true in trues:
                if true in pred_mix:
                    return True
            return False
        except Exception as e:
            print (e)
            return False
    
    def reconstruct_actions(self, template_roles, pred_idxs, recorded_actions):
        pred_actions = []
        for k, v in template_roles.items():
            for pred in v:
                if k != 'R':
                    pred_mech = '%s_%s_%s' % (k, pred_idxs[pred[0]], pred_idxs[pred[1]])
                    pred_actions.append('%s_%s_%s' % (k, pred_idxs[pred[0]], pred_idxs[pred[1]]))
                else:
                    pred_actions.append('%s_%s' % (k, pred_idxs[pred[0]]))
        return pred_actions
    
    def recursive_match(self, preds, trues, template_H_C, n_required_idx):
        matched_idxs = [{}]
        for edit_type in preds:
            if len(trues[edit_type]) == 0:
                continue
            if edit_type == 'R' and self.has_small_fragment and len(preds[edit_type]) == 0:
                continue
            if bidirect_len(preds[edit_type]) < bidirect_len(trues[edit_type]):
                return []
            new_matched_idxs = []
            for matched_idx in matched_idxs:
                matched_idx = match_each(preds[edit_type], trues[edit_type], matched_idx)
                new_matched_idxs += matched_idx
            matched_idxs = new_matched_idxs
        matched_idxs = list(map(dict, set(tuple(sorted(d.items())) for d in matched_idxs if len(d) >= n_required_idx)))
        outputs = []
        for matched_id in matched_idxs:
            recorded_actions = self.template_scores[template_H_C]
            try:
                pred_actions = self.reconstruct_actions(trues, matched_id, recorded_actions)
                if sum([action not in recorded_actions for action in pred_actions]) > 0:
                    continue
                else:
                    outputs.append([matched_id, pred_actions])
            except Exception as e:
                if self.verbose:
                    print (e)
        return outputs

    def collect(self, template, H_code, C_code, pred_action, pred_idx, score):
        correct = False
        template_H_C = '%s_%s_%s' % (template, H_code, C_code)

        template_info = self.templates_info[template_H_C]
        H_change = template_info['change_H']
        C_change = template_info['change_C']
        template_actions = template_info['edit_site']
        n_required_idx = len(set([atom for temp_action, bonds in template_actions.items() for bond in bonds for atom in bond if temp_action != 'R']))
        change_bond_only = len(template_actions['A']) + len(template_actions['B']) + len(template_actions['R']) == 0
        if change_bond_only:
            for i in pred_idx:
                self.template_scores[template_H_C][i] = score
        else:
            if pred_action != 'R':
                pred_mech = '%s_%s_%s' % (pred_action, pred_idx[0], pred_idx[1])
                pred_mech_inv = '%s_%s_%s' % (pred_action, pred_idx[1], pred_idx[0])
                if pred_mech not in self.template_scores[template_H_C]:
                    self.template_scores[template_H_C][pred_mech] = score
                    if pred_action == 'C':
                        self.template_scores[template_H_C][pred_mech_inv] = score
            else:
                pred_mech = '%s_%s' % (pred_action, pred_idx[0])
                if pred_mech not in self.template_scores[template_H_C]:
                    self.template_scores[template_H_C][pred_mech] = score
            
        if n_required_idx >= 6 and change_bond_only:
            n_required_idx -= 2
            
        newly_pred_idxs = []
        if template_H_C not in self.predicted_template:
            if change_bond_only:
                self.predicted_template[template_H_C] = [pred_idx]
                matched_idxs = split_pred_idxs(self.predicted_template[template_H_C])
                if len(matched_idxs) >= n_required_idx:
                    if self.verbose:
                        print ('pred_actions:', matched_idxs)
                        print ('template_actions:', template_actions)
                    correct = self.predict(template_H_C, H_change, C_change, matched_idxs, template_actions, True)
            else:
                self.predicted_template[template_H_C] = [{edit_type:[] for edit_type in template_actions}]
                self.predicted_template[template_H_C][0][pred_action].append(pred_idx)
                pred_idxs = self.predicted_template[template_H_C][0]
                matched_idxs = self.recursive_match(pred_idxs, template_actions, template_H_C, n_required_idx)
                for matched_idx in matched_idxs:
                    correct = self.predict(template_H_C, H_change, C_change, matched_idx, template_actions)
                    if correct:
                        break
                        
        elif change_bond_only:
            self.predicted_template[template_H_C].append(pred_idx)
            matched_idxs = split_pred_idxs(self.predicted_template[template_H_C])
            if len(matched_idxs) >= n_required_idx:
                if self.verbose:
                    print ('pred_actions:', matched_idxs)
                    print ('template_actions:', template_actions)
                correct = self.predict(template_H_C, H_change, C_change, matched_idxs, template_actions, True)
                
        else:
            for pred_idxs in self.predicted_template[template_H_C]: 
                pred_idxs[pred_action].append(pred_idx)
                matched_idxs = self.recursive_match(pred_idxs, template_actions, template_H_C, n_required_idx)
                if self.verbose:
                    print ('pred_actions:', pred_idxs)
                    print ('template_actions:', template_actions)
                    
                for matched_idx in matched_idxs:
                    if self.verbose:
                        print ('matched_idx:', matched_idx)
                    correct = self.predict(template_H_C, H_change, C_change, matched_idx, template_actions)
                    if correct:
                        break
        return correct
    
    def predict(self, template_H_C, H_change, C_change, pred_idxs, template_actions, change_bond_only = False):
        
        if not change_bond_only:
            pred_idxs, pred_actions = pred_idxs
            idx_code = ''.join([str(pred_idxs[k]) for k in pred_idxs.keys()])
            if idx_code in self.used_idx[template_H_C]:
                return False
            else:
                self.used_idx[template_H_C].append(idx_code)
        
        template = template_H_C.split('_')[0]
        try:
            matched_products, fit_temp = apply_template(self, template, H_change, C_change, pred_idxs, change_bond_only, self.verbose)
            if change_bond_only:
                pred_actions = []
                for matched_idx, matched_product in matched_products.items():
                    if matched_product in self.old_predictions:
                        continue
                    for idx in eval(matched_idx).values():
                        if 'C_%s' % idx not in pred_actions and idx in pred_idxs:
                            pred_actions.append('C_%s' % idx) 
            if self.verbose:
                print ('matched_products:', matched_products)
                print (pred_actions)
        
        except KeyboardInterrupt:
            print('Interrupted')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
                
        except Exception as e:
            if self.verbose:
                print (e)
            return False
        correct = False
            
        newly_predicted = []
        for matched_idx, products in matched_products.items():
            if self.products != '':
                products = self.fix_small_product(products)
            for product in products.split('.'):
                if product not in newly_predicted and product not in self.old_predictions:
                    newly_predicted.append(product)
                    self.old_predictions.add(product)
                    
        if self.verbose:
            print ('predicted product(s):', newly_predicted)
            
        if len(newly_predicted) != 0:
            if len(newly_predicted[0]) == 0:
                return correct
            predicted_product = '.'.join(sorted(newly_predicted))
            if change_bond_only:
                score = np.average([self.template_scores[template_H_C][int(action.split('_')[1])] for action in pred_actions])
            else:
                score = np.average([self.template_scores[template_H_C][action] for action in pred_actions])
            correct = self.structure_match(predicted_product, self.products)
            if predicted_product not in self.predictions:
                self.predictions[predicted_product] = {'template':fit_temp, 'pred_actions': pred_actions, 'pred_idx':pred_idxs, 'score':score, 'correct':correct}
        return correct
    
    