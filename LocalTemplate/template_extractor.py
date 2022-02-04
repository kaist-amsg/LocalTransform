'''
This python script is modified from rdchiral template extractor 
https://github.com/connorcoley/rdchiral/blob/master/rdchiral/template_extractor.py
'''
import re
from numpy.random import shuffle
from collections import defaultdict
from pprint import pprint 
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import ChiralType

from .template_extract_utils import *
    
default_setting = {'verbose': False, 'use_stereo': False, 'use_symbol': False, 'max_unmap': 5, 'retro': False, 'remote': True}

def set_extractor(setting):
    global VERBOSE,  USE_STEREOCHEMISTRY, USE_ATOM_SYMBOL, CONNECT_ATOMS, MAXIMUM_NUMBER_UNMAPPED_PRODUCT_ATOMS, RETRO, REMOTE
    VERBOSE = setting['verbose'] # False
    USE_STEREOCHEMISTRY = setting['use_stereo'] # False
    USE_ATOM_SYMBOL = setting['use_symbol'] # False
    MAXIMUM_NUMBER_UNMAPPED_PRODUCT_ATOMS = setting['max_unmap'] # 5
    RETRO = setting['retro']
    REMOTE = setting['remote']
    return

def mols_from_smiles_list(all_smiles, product_maps = []):
    '''Given a list of smiles strings, this function creates rdkit
    molecules'''
    mols = []
    for smiles in all_smiles:
        if not smiles: continue
        mol = Chem.MolFromSmiles(smiles)
        [atom.SetAtomMapNum(0) for atom in mol.GetAtoms() if atom.GetAtomMapNum() not in product_maps]
        mols.append(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))
    return sorted(mols, key= lambda m: m.GetNumAtoms(), reverse=True)

def replace_deuterated(smi):
    return re.sub('\[2H\]', r'[H]', smi)

def clear_mapnum(mol):
    [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms() if a.HasProp('molAtomMapNumber')]
    return mol

def get_tagged_atoms_from_mols(mols):
    '''Takes a list of RDKit molecules and returns total list of
    atoms and their tags'''
    atoms = []
    atom_tags = []
    for mol in mols:
        new_atoms, new_atom_tags = get_tagged_atoms_from_mol(mol)
        atoms += new_atoms 
        atom_tags += new_atom_tags
    return atoms, atom_tags

def get_tagged_atoms_from_mol(mol):
    '''Takes an RDKit molecule and returns list of tagged atoms and their
    corresponding numbers'''
    atoms = []
    atom_tags = []
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atoms.append(atom)
            atom_tags.append(str(atom.GetProp('molAtomMapNumber')))
    return atoms, atom_tags

def atoms_are_different(atom1, atom2): 
    '''Compares two RDKit atoms based on basic properties'''

    if atom1.GetAtomicNum() != atom2.GetAtomicNum(): return True # must be true for atom mapping
    
    if atom1.GetNumRadicalElectrons() != atom2.GetNumRadicalElectrons(): return True
    if REMOTE:
        if atom1.GetFormalCharge() != atom2.GetFormalCharge(): return True
        # may be wrong information due to wrong atom mapping
        if atom1.GetTotalNumHs() != atom2.GetTotalNumHs(): return True
        
    # add or break bonds
    if atom_neighbors(atom1) != atom_neighbors(atom2): return True 
    
    # change bonds
    bonds1 = sorted([bond_to_smarts(bond) for bond in atom1.GetBonds()]) 
    bonds2 = sorted([bond_to_smarts(bond) for bond in atom2.GetBonds()]) 
    if bonds1 != bonds2: return True

    return False

def find_map_num(mol, mapnum):
    return [(a.GetIdx(), a) for a in mol.GetAtoms() if a.HasProp('molAtomMapNumber') 
         and a.GetProp('molAtomMapNumber') == str(mapnum)][0]

def get_tetrahedral_atoms(reactants, products):
    tetrahedral_atoms = []
    for reactant in reactants:
        for ar in reactant.GetAtoms():
            if not ar.HasProp('molAtomMapNumber'):
                continue
            atom_tag = ar.GetProp('molAtomMapNumber')
            ir = ar.GetIdx()
            for product in products:
                try:
                    (ip, ap) = find_map_num(product, atom_tag)
                    if ar.GetChiralTag() != ChiralType.CHI_UNSPECIFIED or\
                            ap.GetChiralTag() != ChiralType.CHI_UNSPECIFIED:
                        tetrahedral_atoms.append((atom_tag, ar, ap))
                except IndexError:
                    pass
    return tetrahedral_atoms

def set_isotope_to_equal_mapnum(mol):
    for a in mol.GetAtoms():
        if a.HasProp('molAtomMapNumber'):
            a.SetIsotope(int(a.GetProp('molAtomMapNumber')))
            
def get_frag_around_tetrahedral_center(mol, idx):
    '''Builds a MolFragment using neighbors of a tetrahedral atom,
    where the molecule has already been updated to include isotopes'''
    ids_to_include = [idx]
    for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
        ids_to_include.append(neighbor.GetIdx())
    symbols = ['[{}{}]'.format(a.GetIsotope(), a.GetSymbol()) if a.GetIsotope() != 0\
               else '[#{}]'.format(a.GetAtomicNum()) for a in mol.GetAtoms()]
    return Chem.MolFragmentToSmiles(mol, ids_to_include, isomericSmiles=True,
                                   atomSymbols=symbols, allBondsExplicit=True,
                                   allHsExplicit=True)
            
def check_tetrahedral_centers_equivalent(atom1, atom2):
    '''Checks to see if tetrahedral centers are equivalent in
    chirality, ignoring the ChiralTag. Owning molecules of the
    input atoms must have been Isotope-mapped'''
    atom1_frag = get_frag_around_tetrahedral_center(atom1.GetOwningMol(), atom1.GetIdx())
    atom1_neighborhood = Chem.MolFromSmiles(atom1_frag, sanitize=False)
    for matched_ids in atom2.GetOwningMol().GetSubstructMatches(atom1_neighborhood, useChirality=True):
        if atom2.GetIdx() in matched_ids:
            return True
    return False

def clear_isotope(mol):
    [a.SetIsotope(0) for a in mol.GetAtoms()]

def get_changed_atoms(reactants, products):
    '''Looks at mapped atoms in a reaction and determines which ones changed'''

    err = 0
    prod_atoms, prod_atom_tags = get_tagged_atoms_from_mols(products)

    if VERBOSE: print('Products contain {} tagged atoms'.format(len(prod_atoms)))
    if VERBOSE: print('Products contain {} unique atom numbers'.format(len(set(prod_atom_tags))))

    reac_atoms, reac_atom_tags = get_tagged_atoms_from_mols(reactants)
    if len(set(prod_atom_tags)) != len(set(reac_atom_tags)):
        if VERBOSE: print('warning: different atom tags appear in reactants and products')
        #err = 1 # okay for Reaxys, since Reaxys creates mass
    if len(prod_atoms) != len(reac_atoms):
        if VERBOSE: print('warning: total number of tagged atoms differ, stoichometry != 1?')
        #err = 1

    # Find differences 
    changed_atoms = [] # actual reactant atom species
    changed_atom_tags = [] # atom map numbers of those atoms

    # Product atoms that are different from reactant atom equivalent
    for i, prod_tag in enumerate(prod_atom_tags):
        for j, reac_tag in enumerate(reac_atom_tags):
            if reac_tag != prod_tag: continue
            if reac_tag not in changed_atom_tags: # don't bother comparing if we know this atom changes
                # If atom changed, add
                if atoms_are_different(prod_atoms[i], reac_atoms[j]):
                    changed_atoms.append(reac_atoms[j])
                    changed_atom_tags.append(reac_tag)
                    break
                # If reac_tag appears multiple times, add (need for stoichometry > 1)
                if prod_atom_tags.count(reac_tag) > 1:
                    changed_atoms.append(reac_atoms[j])
                    changed_atom_tags.append(reac_tag)
                    break

    # Reactant atoms that do not appear in product (tagged leaving groups)
    for j, reac_tag in enumerate(reac_atom_tags):
        if reac_tag not in changed_atom_tags:
            if reac_tag not in prod_atom_tags:
                changed_atoms.append(reac_atoms[j])
                changed_atom_tags.append(reac_tag)

    # Atoms that change CHIRALITY (just tetrahedral for now...)
    if USE_STEREOCHEMISTRY:
        tetra_atoms = get_tetrahedral_atoms(reactants, products)
        if VERBOSE:
            print('Found {} atom-mapped tetrahedral atoms that have chirality specified at least partially'.format(len(tetra_atoms)))
        [set_isotope_to_equal_mapnum(reactant) for reactant in reactants]
        [set_isotope_to_equal_mapnum(product) for product in products]
        for (atom_tag, ar, ap) in tetra_atoms:
            if VERBOSE: 
                print('For atom tag {}'.format(atom_tag))
                print('    reactant: {}'.format(ar.GetChiralTag()))
                print('    product:  {}'.format(ap.GetChiralTag()))
            if atom_tag in changed_atom_tags:
                if VERBOSE:
                    print('-> atoms have changed (by more than just chirality!)')
            else:
                unchanged = check_tetrahedral_centers_equivalent(ar, ap) and \
                        ChiralType.CHI_UNSPECIFIED not in [ar.GetChiralTag(), ap.GetChiralTag()]
                if unchanged:
                    if VERBOSE: 
                        print('-> atoms confirmed to have same chirality, no change')
                else:
                    if VERBOSE:
                        print('-> atom changed chirality!!')
                    # Make sure chiral change is next to the reaction center and not
                    # a random specifidation (must be CONNECTED to a changed atom)
                    tetra_adj_to_rxn = False
                    for neighbor in ap.GetNeighbors():
                        if neighbor.HasProp('molAtomMapNumber'):
                            if neighbor.GetProp('molAtomMapNumber') in changed_atom_tags:
                                tetra_adj_to_rxn = True
                                break
                    if tetra_adj_to_rxn:
                        if VERBOSE:
                            print('-> atom adj to reaction center, now included')
                        changed_atom_tags.append(atom_tag)
                        changed_atoms.append(ar)
                    else:
                        if VERBOSE:
                            print('-> adj far from reaction center, not including')
    [clear_isotope(reactant) for reactant in reactants]
    [clear_isotope(product) for product in products]


    if VERBOSE: 
        print('{} tagged atoms in reactants change 1-atom properties'.format(len(changed_atom_tags)))
        for smarts in [atom.GetSmarts() for atom in changed_atoms]:
            print('  {}'.format(smarts))
    
    return changed_atoms, changed_atom_tags, err

def convert_atom_to_wildcard(atom):
    '''This function takes an RDKit atom and turns it into a wildcard 
    using heuristic generalization rules. This function should be used
    when candidate atoms are used to extend the reaction core for higher
    generalizability'''

    # Is this a terminal atom? We can tell if the degree is one
    if atom.GetDegree() == 1:
        symbol = '[' + atom.GetSymbol() + ';D1;H{}'.format(atom.GetTotalNumHs())
        if atom.GetFormalCharge() != 0:
            charges = re.search('([-+]+[1-9]?)', atom.GetSmarts())
            symbol = symbol.replace(';D1', ';{};D1'.format(charges.group()))
    
    else:
        # Initialize
        symbol = '['

        # Add atom primitive - atomic num and aromaticity (don't use COMPLETE wildcards)
        if atom.GetAtomicNum() != 6:
            symbol += '#{};'.format(atom.GetAtomicNum())
            if atom.GetIsAromatic():
                symbol += 'a;'
        elif atom.GetIsAromatic():
            symbol += 'c;'
        else:
            symbol += 'C;'

        # Charge is important
        if atom.GetFormalCharge() != 0:
            charges = re.search('([-+]+[1-9]?)', atom.GetSmarts())
            if charges: symbol += charges.group() + ';'

        # Strip extra semicolon
        if symbol[-1] == ';': symbol = symbol[:-1]

    # Close with label or with bracket
    label = re.search('\:[0-9]+\]', atom.GetSmarts())
    if label: 
        symbol += label.group()
    else:
        symbol += ']'

    if VERBOSE: 
        if symbol != atom.GetSmarts():
            print('Improved generality of atom SMARTS {} -> {}'.format(atom.GetSmarts(), symbol))

    return symbol

def template_scorer(template, atom_dict):
    score = 0
    for s, b in enumerate(['-', ':', '=', '#']):
        score += template.count(b) * (s+1)
    for n in re.findall('\:([0-9]+)\]', template):
        score += 0.1*atom_dict[n]['charge'] + 0.01*atom_dict[n]['Hs']
    return score
    
def inverse_template(template):
    def score_bonds(bonds):
        bond_dict = {b:str(i+1) for i,b in enumerate(['-', ':', '=', '#'])}
        return eval(''.join([bond_dict[b] for b in bonds]))
    if ']1' in template:
        ring_template = True
    else:
        ring_template = False
#     bonds1 = re.findall('([-=#])', template)
    bonds1 = re.findall(r']([-=#:])|]1([-=#:])', template)
    bonds1 = [sorted(bond)[1] for bond in bonds1]
    bonds2 = bonds1[::-1]
    if len(bonds1) == 0 or ')' in template:
        return template
    if score_bonds(bonds1) <= score_bonds(bonds2):
        return template
    
    labels = re.findall(r'\[.*?]', template)[::-1]
    inv_template = labels[0]
    for i in range(len(bonds2)):
        if ring_template:
            if i == 0:
                inv_template += '1'
            if i+1 == len(labels):
                inv_template += bonds2[0]
                inv_template += '1' 
            else:
                inv_template += bonds2[i+1]
                inv_template += labels[i+1]
        else:
            inv_template += bonds2[i]
            inv_template += labels[i+1]
    return inv_template

def canonicalize_smarts(smarts):
    if USE_ATOM_SYMBOL:
        return smarts
    preserved_info = {'[#0:%s]' % a.split(':')[-1].split(']')[0]: a for a in re.findall(r"\[.*?]", smarts)}
    try:
        smiles = Chem.MolToSmiles(Chem.MolFromSmarts(smarts))
        smarts_ = Chem.MolToSmarts(Chem.MolFromSmiles(smiles))
    except:
        return smarts
    if '(' not in smarts_:
        smarts = smarts_
        for k, v in preserved_info.items():
            smarts = smarts.replace(k, v)
    return smarts

def sort_template(transform, atom_dict):
    transform = transform.split('>>')[0][1:-1].replace(').(', '.') + \
        '>>' + transform.split('>>')[1][1:-1].replace(').(', '.')
    templates = []
    for tt in transform.split('>>'):
        ts = []
        for smarts in sorted(tt.split('.'), key=lambda s: template_scorer(s, atom_dict)):
            ts.append(inverse_template(canonicalize_smarts(smarts)))
#             ts.append(canonicalize_smarts(smarts))
        templates.append('.'.join(ts))
    return '>>'.join(templates)

def permutations(template):
    labels = re.findall('\:([0-9]+)\]', template)
    if len(labels) == 1 or '(' in template:
        return [labels]
    charges = re.findall('\;(.+?[0-9]+)\:', template)
    bonds = re.findall('\]([-=#:])\[', template)
    if ''.join(bonds) != ''.join(bonds[::-1]) or ''.join(charges) != ''.join(charges[::-1]):
        return [labels]
    return [labels,  labels[::-1]]
        
def enumerate_mapping(transform):
    for i, templates in enumerate(transform.split('>>')):
        grow_template = None
        for template in templates.split('.'):
            pert_template = permutations(template)
            if grow_template == None:
                grow_template = pert_template
            else:
                growed_template = []
                for t in grow_template:
                    for p in pert_template:
                        growed_template.append(t + p)
                grow_template = growed_template
        if i == 0:
            r_permutes = grow_template
        else:
            p_permutes = grow_template
                
    t_permutes = []
    for r in r_permutes:
        for p in p_permutes:
                t_permutes.append(r+p)
    return t_permutes

def reassign_atom_mapping(transform, atom_dict):
    if not RETRO:
        transform = '>>'.join(transform.split('>>')[::-1])
    transform = sort_template(transform, atom_dict)
    p_labels = enumerate_mapping(transform)
    templates = set()
    templates_sort = {}
    replacement_dicts = {}
    for all_labels in p_labels:
    # Define list of replacements which matches all_labels *IN ORDER*
        replacements = []
        replacement_dict = {}
        counter = 1
        for label in all_labels: # keep in order! this is important
            if label not in replacement_dict:
                replacement_dict[label] = str(counter)
                counter += 1
            replacements.append(replacement_dict[label])
            
        # Perform replacements in order
        transform_newmaps = re.sub('\:[0-9]+\]', lambda match: (':' + replacements.pop(0) + ']'), transform)
        if not RETRO:
            transform_newmaps = '>>'.join(transform_newmaps.split('>>')[::-1])
        templates.add(transform_newmaps)
        templates_sort[transform_newmaps] = ''.join(re.findall('\:([0-9]+)\]', transform_newmaps))
        replacement_dicts[transform_newmaps] = replacement_dict
    transform_newmaps = sorted(list(templates), key = lambda t:templates_sort[t])[0]
    replacement_dict = replacement_dicts[transform_newmaps]
    return transform_newmaps, replacement_dict

def get_strict_smarts_for_atom(atom):
    '''
    For an RDkit atom object, generate a SMARTS pattern that
    matches the atom as strictly as possible
    '''
    symbol = atom.GetSmarts()
    if not USE_ATOM_SYMBOL:
        symbol = '[A:%s]' % (atom.GetAtomMapNum())
    
    elif 'H' in symbol and 'Hg' not in symbol:
        symbol = symbol.split('H')[0].split('[')[1]
        symbol = '[%s:%s]' % (symbol, atom.GetAtomMapNum())
    
    if atom.GetSymbol() == 'H':
        symbol = '[#1]'

    if '[' not in symbol:
            symbol = '[' + symbol + ']'

    # Explicit stereochemistry - *before* H
    if USE_STEREOCHEMISTRY:
        if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            if '@' not in symbol:
                # Be explicit when there is a tetrahedral chiral tag
                if atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
                    tag = '@'
                elif atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
                    tag = '@@'
                if ':' in symbol:
                    symbol = symbol.replace(':', ';{}:'.format(tag))
                else:
                    symbol = symbol.replace(']', ';{}]'.format(tag))
       
    return symbol

def get_fragments_for_changed_atoms(mols, changed_atom_tags, category = 'reactant'):
    fragments = ''
    mols_changed = []
    for mol in mols:
        # Initialize list of replacement symbols (updated during expansion)
        symbol_replacements = []

        # Build list of atoms to use
        atoms_to_use = []
        for atom in mol.GetAtoms():
            # Check self (only tagged atoms)
            if ':' in atom.GetSmarts():
                if atom.GetSmarts().split(':')[1][:-1] in changed_atom_tags:
                    atoms_to_use.append(atom.GetIdx())
                    symbol = get_strict_smarts_for_atom(atom)
                    if symbol != atom.GetSmarts():
                        symbol_replacements.append((atom.GetIdx(), symbol))
                    continue

        # Fully define leaving groups and this molecule participates?
        if category == 'reactant' and len(atoms_to_use) > 0 and RETRO:
            for atom in mol.GetAtoms():
                if not atom.HasProp('molAtomMapNumber'): # LG in reactant is also mapped in foward synthesis
                    atoms_to_use.append(atom.GetIdx())
                    
        # Define new symbols based on symbol_replacements
        symbols = [atom.GetSmarts() for atom in mol.GetAtoms()]
        for (i, symbol) in symbol_replacements:
            symbols[i] = symbol
        if not atoms_to_use: 
            continue
        
        # Keep flipping stereocenters until we are happy...
        # this is a sloppy fix during extraction to achieve consistency
        tetra_consistent = False
        num_tetra_flips = 0
        while not tetra_consistent and num_tetra_flips < 100:
            mol_copy = deepcopy(mol)
            [x.ClearProp('molAtomMapNumber') for x in mol_copy.GetAtoms()]   
            this_fragment = AllChem.MolFragmentToSmiles(mol_copy, atoms_to_use, 
                atomSymbols=symbols, allHsExplicit=True, 
                isomericSmiles=USE_STEREOCHEMISTRY, allBondsExplicit=True)

            # Figure out what atom maps are tetrahedral centers
            # Set isotopes to make sure we're getting the *exact* match we want
            this_fragment_mol = AllChem.MolFromSmarts(this_fragment)
            tetra_map_nums = []
            for atom in this_fragment_mol.GetAtoms():
                if atom.HasProp('molAtomMapNumber'):
                    atom.SetIsotope(int(atom.GetProp('molAtomMapNumber')))
                    if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
                        tetra_map_nums.append(atom.GetProp('molAtomMapNumber'))
            map_to_id = {}
            for atom in mol.GetAtoms():
                if atom.HasProp('molAtomMapNumber'):
                    atom.SetIsotope(int(atom.GetProp('molAtomMapNumber')))
                    map_to_id[atom.GetProp('molAtomMapNumber')] = atom.GetIdx()
            
            # Look for matches
            tetra_consistent = True
            all_matched_ids = []
            
            # skip substructure matching if there are a lot of fragments
            # this can help prevent GetSubstructMatches from hanging 
            frag_smi = Chem.MolToSmiles(this_fragment_mol)
            if frag_smi.count('.') > 5:
                break
            
            for matched_ids in mol.GetSubstructMatches(this_fragment_mol, useChirality=True):
                all_matched_ids.extend(matched_ids)
            shuffle(tetra_map_nums)
            for tetra_map_num in tetra_map_nums:
                if VERBOSE: print('Checking consistency of tetrahedral {}'.format(tetra_map_num))
                #print('Using fragment {}'.format(Chem.MolToSmarts(this_fragment_mol, True)))
                if map_to_id[tetra_map_num] not in all_matched_ids:
                    tetra_consistent = False
                    if VERBOSE: print('@@@@@@@@@@@ FRAGMENT DOES NOT MATCH PARENT MOL @@@@@@@@@@@@@@')
                    if VERBOSE: print('@@@@@@@@@@@ FLIPPING CHIRALITY SYMBOL NOW      @@@@@@@@@@@@@@')
                    prevsymbol = symbols[map_to_id[tetra_map_num]]
                    if '@@' in prevsymbol:
                        symbol = prevsymbol.replace('@@', '@')
                    elif '@' in prevsymbol:
                        symbol = prevsymbol.replace('@', '@@')
                    else:
                        raise ValueError('Need to modify symbol of tetra atom without @ or @@??')
                    symbols[map_to_id[tetra_map_num]] = symbol
                    num_tetra_flips += 1
                    # IMPORTANT: only flip one at a time
                    break 

            # Clear isotopes
            for atom in mol.GetAtoms():
                atom.SetIsotope(0)
        
        if not tetra_consistent:
            raise ValueError('Could not find consistent tetrahedral mapping, {} centers'.format(len(tetra_map_nums)))

        fragments += '(' + this_fragment + ').'
        mols_changed.append(Chem.MolToSmiles(clear_mapnum(Chem.MolFromSmiles(Chem.MolToSmiles(mol, True))), True))

    # auxiliary template information: is this an intramolecular reaction or dimerization?
    intra_only = (1 == len(mols_changed))
    dimer_only = (1 == len(set(mols_changed))) and (len(mols_changed) == 2)
    
    return fragments[:-1], intra_only, dimer_only

def canonicalize_transform(transform, atom_dict):
    '''This function takes an atom-mapped SMARTS transform and
    converts it to a canonical form by, if nececssary, rearranging
    the order of reactant and product templates and reassigning
    atom maps.'''

    transform_reordered = '>>'.join([canonicalize_template(x) for x in transform.split('>>')])
    return reassign_atom_mapping(transform_reordered, atom_dict)

def canonicalize_template(template):
    '''This function takes one-half of a template SMARTS string 
    (i.e., reactants or products) and re-orders them based on
    an equivalent string without atom mapping.'''

    # Strip labels to get sort orders
    template_nolabels = re.sub('\:[0-9]+\]', ']', template)

    # Split into separate molecules *WITHOUT wrapper parentheses*
    template_nolabels_mols = template_nolabels[1:-1].split(').(')
    template_mols          = template[1:-1].split(').(')

    # Split into fragments within those molecules
    for i in range(len(template_mols)):
        nolabel_mol_frags = template_nolabels_mols[i].split('.')
        mol_frags         = template_mols[i].split('.')

        # Get sort order within molecule, defined WITHOUT labels
        sortorder = [j[0] for j in sorted(enumerate(nolabel_mol_frags), key = lambda x:x[1])]

        # Apply sorting and merge list back into overall mol fragment
        template_nolabels_mols[i] = '.'.join([nolabel_mol_frags[j] for j in sortorder])
        template_mols[i]          = '.'.join([mol_frags[j] for j in sortorder])

    # Get sort order between molecules, defined WITHOUT labels
    sortorder = [j[0] for j in sorted(enumerate(template_nolabels_mols), key = lambda x:x[1])]

    # Apply sorting and merge list back into overall transform
    template = '(' + ').('.join([template_mols[i] for i in sortorder]) + ')'

    return template

def bond_to_smarts(bond):
    '''This function takes an RDKit bond and creates a label describing
    the most important attributes'''
    a1_label = str(bond.GetBeginAtom().GetAtomicNum())
    a2_label = str(bond.GetEndAtom().GetAtomicNum())
    if bond.GetBeginAtom().HasProp('molAtomMapNumber'):
        a1_label += bond.GetBeginAtom().GetProp('molAtomMapNumber')
    if bond.GetEndAtom().HasProp('molAtomMapNumber'):
        a2_label += bond.GetEndAtom().GetProp('molAtomMapNumber')
    atoms = sorted([a1_label, a2_label])
    bond_smarts = bond.GetSmarts()
    if bond_smarts == '':
        bond_smarts = '-'
    
    return '{}{}{}'.format(atoms[0], bond_smarts, atoms[1])


def extend_atom_tag(reactant, max_num):
    untagged_neighbors = []
    is_reagent = True
    
    # clean atommap
    for atom in reactant.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            pass
        else:
            is_reagent = False
    
    if is_reagent:
        return is_reagent, max_num
    
    # tag the untag neighbors  
    for atom in reactant.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            continue
        for n_atom in atom.GetNeighbors():
            if n_atom.GetAtomMapNum() == 0:
                untagged_neighbors.append(n_atom.GetIdx())
                
    for idx in untagged_neighbors:
        max_num += 1
        atom = reactant.GetAtomWithIdx(idx)
        atom.SetAtomMapNum(max_num)
    return is_reagent, max_num

def extract_from_reaction(reaction, setting = default_setting):
    set_extractor(setting)
    if type(reaction) == type('string'):
        reaction = {'reactants': reaction.split('>>')[0], 'products': reaction.split('>>')[1], '_id' : 0}
        
    product_maps = [atom.GetAtomMapNum() for atom in Chem.MolFromSmiles(reaction['products']).GetAtoms()]
    products = mols_from_smiles_list(replace_deuterated(reaction['products']).split('.'), product_maps)
    reactants_ = mols_from_smiles_list(replace_deuterated(reaction['reactants']).split('.'), product_maps)

    max_num = max(product_maps)
    reagents = []
    reactants = []
    for reactant in reactants_:
        is_reagent, max_num = extend_atom_tag(reactant, max_num)
        if is_reagent:
            reagents.append(Chem.MolToSmiles(reactant))
        elif Chem.MolToSmiles(reactant) in reaction['products'].split('.'):
            [atom.SetAtomMapNum(0) for atom in reactant.GetAtoms()]
            reagents.append(Chem.MolToSmiles(reactant))
        else:
            reactants.append(reactant)
        
    # if rdkit cant understand molecule, return
    if None in reactants: return {'reaction_id': reaction['_id']} 
    if None in products: return {'reaction_id': reaction['_id']}
    # try to sanitize molecules
    try:
        for i in range(len(reactants)):
            reactants[i] = AllChem.RemoveHs(reactants[i]) # *might* not be safe
        for i in range(len(products)):
            products[i] = AllChem.RemoveHs(products[i]) # *might* not be safe
        [Chem.SanitizeMol(mol) for mol in reactants + products] # redundant w/ RemoveHs
        [mol.UpdatePropertyCache() for mol in reactants + products]
    except Exception as e:
        print(e)
        print('Could not load SMILES or sanitize')
        print('ID: {}'.format(reaction['_id']))
        return {'reaction_id': reaction['_id']}

    if None in reactants + products:
        print('Could not parse all molecules in reaction, skipping')
        print('ID: {}'.format(reaction['_id']))
        return {'reaction_id': reaction['_id']}

    
    # Calculate changed atoms
    changed_atoms, changed_atom_tags, err = get_changed_atoms(reactants, products)
            
    if err: 
        if VERBOSE:
            print('Could not get changed atoms')
            print('ID: {}'.format(reaction['_id']))
        return
    if not changed_atom_tags:
        if VERBOSE:
            print('No atoms changed?')
            print('ID: {}'.format(reaction['_id']))
        return {'reaction_id': reaction['_id']}

    try:
        reactant_fragments, intra_only, dimer_only = get_fragments_for_changed_atoms(reactants, changed_atom_tags)
        product_fragments, _, _  = get_fragments_for_changed_atoms(products, changed_atom_tags, 'product')

    except ValueError as e:
        if VERBOSE:
            print(e)
            print(reaction['_id'])
        return {'reaction_id': reaction['_id']}
    
    # Put together and canonicalize (as best as possible)
    rxn_string = reactant_fragments + '>>' + product_fragments
    atom_dict = {str(atom.GetAtomMapNum()): {'charge': atom.GetFormalCharge(), 'Hs': atom.GetNumExplicitHs()} for atom in changed_atoms}
    
    rxn_canonical, replacement_dict = canonicalize_transform(rxn_string, atom_dict)
    reactants_string = rxn_canonical.split('>>')[0]
    products_string  = rxn_canonical.split('>>')[1]
    products_smiles = '.'.join([Chem.MolToSmiles(p) for p in products])
    reactants_smiles = '.'.join([Chem.MolToSmiles(r) for r in reactants])
    try:
        products_string = canonicalize_smarts(products_string)
        reactants_string = canonicalize_smarts(reactants_string)
    except:
        pass
      
    if RETRO:
        canonical_template = products_string + '>>' + reactants_string
    else:
        canonical_template = reactants_string + '>>' + products_string
    edits, H_change, Charge_change = match_label(reactants_smiles, products_smiles, replacement_dict, changed_atom_tags, retro = RETRO, remote = REMOTE)    
    
    rxn = AllChem.ReactionFromSmarts(canonical_template)
    if rxn.Validate()[1] != 0: 
        print('Could not validate reaction successfully')
        print('ID: {}'.format(reaction['_id']))
        print('canonical_template: {}'.format(canonical_template))
        print (reaction['reactants'] + '>>' + reaction['products'])
        if VERBOSE: raw_input('Pausing...')
        return {'reaction_id': reaction['_id']}

    results = {
    'products': products_smiles,
    'reactants': reactants_smiles,
    'necessary_reagent': reagents,
    'reaction_smarts': canonical_template,
    'intra_only': intra_only,
    'dimer_only': dimer_only,
    'reaction_id': reaction['_id'],
    'replacement_dict': replacement_dict,
    'change_atoms': changed_atom_tags,
    'edits': edits,
    'H_change': H_change,
    'Charge_change': Charge_change
    }
    return results
