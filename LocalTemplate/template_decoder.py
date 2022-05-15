import os, re, copy
import pandas as pd
from collections import defaultdict

import rdkit
from rdkit import Chem, RDLogger 
from rdkit.Chem import rdChemReactions
from rdkit.Chem.rdchem import ChiralType
RDLogger.DisableLog('rdApp.*')
metals = ['Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'Zn', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
chiral_type_map = {ChiralType.CHI_UNSPECIFIED : -1, ChiralType.CHI_TETRAHEDRAL_CW: 1, ChiralType.CHI_TETRAHEDRAL_CCW: 2}
chiral_type_map_inv = {v:k for k, v in chiral_type_map.items()}

def fit_template_with_mol(smiles, transform, pred_idxs):
    mol = Chem.MolFromSmiles(smiles)
    fit_transform = []
    for i, template in enumerate(transform.split('>>')):
        for k, v in pred_idxs.items():
            a = list(re.findall('\[\*:%s\]' % k, template))
            if len(a) == 0:
                continue
            else:
                a = a[0]
            atom = mol.GetAtomWithIdx(v)
            atom_symbol = atom.GetSymbol()
            if atom.GetIsAromatic():
                atom_symbol = atom_symbol.lower()
            b = a.replace('*', atom_symbol)
            template = template.replace(a, b)
        fit_transform.append(template)
    return '>>'.join(fit_transform)

def remove_inreactive_fragment(smiles, transform, reactant_map):
    transform = transform.replace(').(', ')).((')
    new_smiles = []
    new_transform = []
    for s, t in zip(smiles.split('.'), transform.split(').(')):
        if '*' not in t:
            new_smiles.append(s)
            new_transform.append(t)
    return '.'.join(new_smiles), '.'.join(new_transform)

def get_atom_idx_in_mol(smiles):
    all_idx = defaultdict(int)
    for s in smiles.split('.'):
        m = Chem.MolFromSmiles(s)
        for atom in m.GetAtoms():
            all_idx[atom.GetIdx()] += 1
    return all_idx

def molsfromsmiles(smiles):
    mols = []
    count = 0
    for i, s in enumerate(smiles.split('.')):
        m = Chem.MolFromSmiles(s)
        [atom.SetUnsignedProp('_original_idx', count + atom.GetIdx()) for atom in m.GetAtoms()]
        count += m.GetNumAtoms()
        mols.append(m)
    return mols

def fix_product_CHS(reactant_smiles, products, matched_idx, conf_changes):
    fixed_mols = []
    reactants = Chem.MolFromSmiles(reactant_smiles)
    for mol in products:
        for atom in mol.GetAtoms():
            if atom.HasProp('old_mapno'):
                mapno = int(atom.GetProp('old_mapno'))
                reactant_atom = reactants.GetAtomWithIdx(matched_idx[mapno])
                H_before = reactant_atom.GetNumExplicitHs()
                C_before = reactant_atom.GetFormalCharge()
                S_before = chiral_type_map[reactant_atom.GetChiralTag()]
                if mapno not in conf_changes['H']:
                    if atom.GetSymbol() in metals:
                        H_after = H_before
                        C_after = C_before + 1
                    else:
                        H_after = H_before + 1
                        C_after = C_before
                else:
                    H_after = H_before + conf_changes['H'][mapno]
                    C_after = C_before + conf_changes['C'][mapno]
                    S_after = conf_changes['S'][mapno]
                if H_after < 0:
                    H_after = 0
                atom.SetNumExplicitHs(H_after)
                atom.SetFormalCharge(C_after)
                if S_after != 0:
                    atom.SetChiralTag(chiral_type_map_inv[S_after])
        fixed_mols.append(mol)
    return tuple(fixed_mols)
    
def run_reaction(smiles, template, pred_idxs, inverse_map, conf_changes, reactant_map, template_map, change_bond_only, verbose):
    matched_products = dict()
    if change_bond_only:
        fit_temp = template
    else:
        fit_temp = fit_template_with_mol(smiles, template, pred_idxs)
    if verbose:
        print ('Fit_template:', fit_temp)
    reaction = rdChemReactions.ReactionFromSmarts(fit_temp)
    products = reaction.RunReactants(molsfromsmiles(smiles))
    for i, product in enumerate(products):
        matched_idx, left_atoms = check_idx_match(product, reactant_map, template_map)
        common_keys = match_subkeys(pred_idxs, matched_idx, left_atoms, change_bond_only)
        try:
            product = fix_product_CHS(smiles, product, matched_idx, conf_changes)
            product_smiles = '.'.join(sorted([demap(r) for r in product]))
        except Exception as e:
            if verbose:
                print (e)
            continue
        if common_keys:
            reaction_center = {k: inverse_map[v] for k, v in common_keys.items()}
            matched_products[str(reaction_center)] = product_smiles
            if verbose:
                print ('reaction_center:', reaction_center, 'product:', product_smiles)
    return matched_products, fit_temp

def check_idx_match(mols, reactant_map, template_map):
    matched_idx = {}
    left_atoms = []
    for mol in mols:
        for atom in mol.GetAtoms():
            if atom.HasProp('old_mapno'):
                mapno = int(atom.GetProp('old_mapno'))
                new_idx = int(atom.GetProp('react_atom_idx'))
                position = template_map[mapno]
                old_idx = reactant_map[position][new_idx]
                matched_idx[mapno] = old_idx
                left_atoms.append(old_idx)
            elif atom.HasProp('_original_idx'):
                left_atoms.append(int(atom.GetProp('_original_idx')))
    return matched_idx, left_atoms

def match_subkeys(dict1, dict2, left_atoms, change_bond_only):
    if change_bond_only: # dict2 should be subset of list1
        if set(dict2.values()).issubset(set(dict1)) or set(dict1).issubset(set(dict2.values())):
            return dict2
        else:
            return False
    if len(dict1) == 0:
        return False
    common_keys = copy.copy(dict1)
    # check removed  atoms
    for k, v in dict1.items():
        if k not in dict2 and v in left_atoms:
            return False
        
    # newly detect atoms
    for k, v in dict2.items():
        if k not in dict1:
            common_keys[k] = v
        elif dict1[k] != dict2[k]:
            return False
    return common_keys
    
def fix_aromatic(mol):
    for bond in mol.GetBonds():
        if not bond.IsInRing():
            bond.SetIsAromatic(False) 
            if str(bond.GetBondType()) == 'AROMATIC':
                bond.SetBondType(Chem.rdchem.BondType.SINGLE)
                
    for atom in mol.GetAtoms():
        if not atom.IsInRing() and atom.GetIsAromatic():
            atom.SetIsAromatic(False)
        if sum([bond.GetIsAromatic() for bond in atom.GetBonds()]) > 0:
            atom.SetIsAromatic(True)
    return

def deradical(mol):
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() != 0:
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + atom.GetNumRadicalElectrons())
            atom.SetNumRadicalElectrons(0)
    return mol

def demap(mol): 
    fix_aromatic(mol)
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
    smi = Chem.MolToSmiles(mol)
    return Chem.MolToSmiles(deradical(Chem.MolFromSmiles(smi)))

def select_right_products(matched_product_list, verbose):
    right_products = []
    for product in matched_product_list:
        try:
            product_smiles = '.'.join(sorted([demap(r) for r in product]))
            if product_smiles not in right_products:
                right_products.append(product_smiles)
        except Exception as e:
            if verbose:
                print (e)
    return right_products
    
# make the template order same with the reactants order
def prepare_template(collector, template, pred_idxs, change_bond_only):
    smiles = collector.reactant
    non_reacts = collector.non_reacts
    
    if not change_bond_only:
        pred_idxs_inv = {v:k for k,v in pred_idxs.items()}
    else:
        pred_idxs_inv = {v:k+1 for k,v in enumerate(pred_idxs)}
    atom_nums = []
    new_smiles = []
    reduced_map = {}
    reactant_map = {}
    current_num = 0
    include_num = 0
    remap_num = 0
    for s in smiles.split('.'):
        m = Chem.MolFromSmiles(s)
        n = [pred_idxs_inv[atom.GetIdx()+current_num] for atom in m.GetAtoms() if atom.GetIdx()+current_num in pred_idxs_inv]
        if len(n) > 0 and s not in non_reacts:
            new_smiles.append(s)
            reactant_map[include_num] = {}
            for atom in m.GetAtoms():
                reduced_map[atom.GetIdx() + current_num] = atom.GetIdx() + remap_num
                reactant_map[include_num][atom.GetIdx()] = atom.GetIdx() + remap_num
            remap_num += m.GetNumAtoms()
            include_num += 1
        if len(s) != 0 and s not in non_reacts:
            atom_nums.append(n)
        current_num += m.GetNumAtoms()
        
    inverse_map = {v:k for k,v in reduced_map.items()}
    if not change_bond_only:
        new_pred_idxs = {k:reduced_map[v] for k, v in pred_idxs.items()}
    else:
        new_pred_idxs = [reduced_map[v] for v in pred_idxs]
    temp_reactants = template.split('>>')[0].split('.')
    temp_nums = {temp: [int(r) for r in re.findall('\:([0-9]+)\]', temp)] for temp in temp_reactants}
    new_temp_reactants = []
    for nums in atom_nums:
        match_temp = False
        fragment_temp = []
        matched_nums = []
        for num in nums:
            if num in matched_nums:
                continue
            for temp, temp_num in temp_nums.items():
                if temp in fragment_temp:
                    continue
                if num in temp_num:
                    fragment_temp.append(temp)
                    matched_nums += temp_num
                    match_temp = True
        if match_temp == False:
            if collector.sep:
                return None, None, None, None
        else:
            new_temp_reactants.append('(%s)' % '.'.join(fragment_temp))
       
    return '.'.join(new_smiles), reactant_map, new_pred_idxs, inverse_map, '%s>>%s' % ('.'.join(new_temp_reactants), '(' + template.split('>>')[1] + ')')

def get_template_position(transform):
        templates = transform.split('>>')[0].split(').(')
        template_map = {}
        for i, template in enumerate(templates):
            for num in template.split(']')[:-1]:
                template_map[int(num.split(':')[-1])] = i
        return template_map

def apply_template(collector, template, conf_changes, pred_idxs, change_bond_only = False, verbose = False):
    reduced_smiles, reactant_map, reduced_pred_idxs, inverse_map, template = prepare_template(collector, template, pred_idxs, change_bond_only)
    if verbose:
        print (collector.non_reacts)
        print (template)
    if not template:
        return dict()
    template = template.replace('A', '*')
    template_map = get_template_position(template)
    if verbose:
        print (reactant_map, template_map)
        print ('template:', template, 'pred_idxs:', reduced_pred_idxs, 'Conf_change:', conf_changes)
    matched_products, fit_temp = run_reaction(reduced_smiles, template, reduced_pred_idxs, inverse_map, conf_changes, reactant_map, template_map, change_bond_only, verbose)

    return matched_products, fit_temp
