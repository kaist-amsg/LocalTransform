import os, sys, re, copy
import pandas as pd
import rdkit 
from rdkit import Chem, RDLogger 
from rdkit.Chem import rdChemReactions
RDLogger.DisableLog('rdApp.*')  
sys.path.append('../')
from LocalTemplate.template_extractor import extract_from_reaction
from Extract_from_train_data import build_template_extractor, get_reaction_template, get_full_template
    

def get_edit_site(smiles):
    mol = Chem.MolFromSmiles(smiles)
    A = [a for a in range(mol.GetNumAtoms())]
    B = []
    for atom in mol.GetAtoms():
        others = []
        bonds = atom.GetBonds()
        for bond in bonds:
            atoms = [bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()]
            other = [a for a in atoms if a != atom.GetIdx()][0]
            others.append(other)
        b = [(atom.GetIdx(), other) for other in sorted(others)]
        B += b
    V = []
    for a in A:
        V += [(a,b) for b in A if a != b and (a,b) not in B]
    return V, B
    
def labeling_dataset(args, split, template_dicts, template_infos, extractor):
    
    if os.path.exists('../data/%s/preprocessed_%s.csv' % (args['dataset'], split)) and args['force'] == False:
        print ('%s data already preprocessed...loaded data!' % split)
        return pd.read_csv('../data/%s/preprocessed_%s.csv' % (args['dataset'], split))
    
    rxns = {}
    with open('../data/%s/%s.txt' % (args['dataset'], split), 'r') as f:
        for i, line in enumerate(f.readlines()):
            rxns[i] = line.split(' ')[0]
    reactants = []
    reagents = []
    products = []
    labels_sep = []
    labels_mix = []
    frequency = []
    success = 0
    
    for i, reaction in rxns.items():
        reactant = reaction.split('>>')[0]
        reagent = ''
        rxn_labels_s = []
        rxn_labels_m = []
        try:
            rxn, result = get_reaction_template(extractor, reaction, i)
            template = result['reaction_smarts']
            reactant = result['reactants']
            product = result['products']
            reagent = '.'.join(result['necessary_reagent'])
            reactants.append(reactant)
            reagents.append(reagent)
            products.append(product)
            
            if len(result['necessary_reagent']) == 0:
                reactant_mix = reactant
            else:
                reactant_mix = '%s.%s' % (reactant, reagent)
            edit_bonds = {edit_type: edit_bond[0] for edit_type, edit_bond in result['edits'].items()}
            H_change, Charge_change, Chiral_change = result['H_change'], result['Charge_change'], result['Chiral_change']
            template = get_full_template(template, H_change, Charge_change, Chiral_change)
            
            if template not in template_infos.keys():
                labels_sep.append(rxn_labels_s)
                labels_mix.append(rxn_labels_m)
                frequency.append(0)
                continue
                
        except KeyboardInterrupt:
            print('Interrupted')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
        except Exception as e:
            print (i, e)
            labels_sep.append(rxn_labels_s)
            labels_mix.append(rxn_labels_m)
            frequency.append(0)
            continue
        
        edit_n = 0
        for edit_type in edit_bonds:
            if edit_type == 'C':
                edit_n += len(edit_bonds[edit_type])/2
            else:
                edit_n += len(edit_bonds[edit_type])
            
            
        if edit_n <= args['max_edit_n']:
            virtual_sites_s, real_sites_s = get_edit_site(reactant)
            virtual_sites_m, real_sites_m = get_edit_site(reactant_mix)
            try:
                success += 1
                for edit_type in edit_bonds:
                    bonds = edit_bonds[edit_type]
                    for bond in bonds:
                        if edit_type != 'A':
                            rxn_labels_s.append(('r', real_sites_s.index(bond), template_dicts['real']['%s_%s' % (template, edit_type)]))
                            rxn_labels_m.append(('r', real_sites_m.index(bond), template_dicts['real']['%s_%s' % (template, edit_type)]))
                        else:
                            rxn_labels_s.append(('v', virtual_sites_s.index(bond), template_dicts['virtual']['%s_%s' % (template, edit_type)]))
                            rxn_labels_m.append(('v', virtual_sites_m.index(bond), template_dicts['virtual']['%s_%s' % (template, edit_type)]))      
                labels_sep.append(rxn_labels_s)
                labels_mix.append(rxn_labels_m)
                frequency.append(template_infos[template]['frequency'])
                        
            except Exception as e:
                print (i, e)
                labels_sep.append(rxn_labels_s)
                labels_mix.append(rxn_labels_m)
                frequency.append(0)
                continue
                
            if i % 100 == 0:
                print ('\r Processing %s %s data..., success %s data (%s/%s)' % (args['dataset'], split, success, i, len(rxns)), end='', flush=True)
        else:
            print ('\nReaction # %s has too many edits (%s)...may be wrong mapping!' % (i, edit_n))
            labels_sep.append(rxn_labels_s)
            labels_mix.append(rxn_labels_m)
            frequency.append(0)
            
    print ('\nDerived tempaltes cover %.3f of %s data reactions' % ((success/len(rxns)), split))
    
    df = pd.DataFrame({'Reactants': reactants, 'Reagents': reagents, 'Products': products, 'Labels_sep': labels_sep, 'Labels_mix': labels_mix, 'Frequency': frequency})
    df.to_csv('../data/%s/preprocessed_%s.csv' % (args['dataset'], split))
    return df
    
    
def combine_preprocessed_data(train_data, val_data, test_data, args):
    train_data['Split'] = ['train'] * len(train_data)
    val_data['Split'] = ['valid'] * len(val_data)
    test_data['Split'] = ['test'] * len(test_data)
    all_data = train_data.append(val_data, ignore_index=True)
    all_data = all_data.append(test_data, ignore_index=True)
    all_data['Mask'] = [int(f>=args['min_template_n']) for f in all_data['Frequency']]
    print ('Valid data size: %s' % len(all_data))
    all_data.to_csv('../data/%s/labeled_data.csv' % args['dataset'], index = None)
    return

def load_templates(args):
    template_dicts = {}
    for site in ['real', 'virtual']:
        template_df = pd.read_csv('../data/%s/%s_templates.csv' % (args['dataset'], site))
        template_dict = {template_df['Template'][i]:template_df['Class'][i] for i in template_df.index}
        print ('loaded %s %s templates' % (len(template_dict), site))
        template_dicts[site] = template_dict
                                          
    template_infos = pd.read_csv('../data/%s/template_infos.csv' % args['dataset'])
    template_infos = {template_infos['Template'][i]: {'edit_site': eval(template_infos['edit_site'][i]), 'change_H': eval(template_infos['change_H'][i]), 'frequency': template_infos['Frequency'][i]} for i in template_infos.index}
    print ('loaded total %s templates' % len(template_infos))
    return template_dicts, template_infos
                                          
if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser('LocalTempalte Preprocessing')
    parser.add_argument('-d', '--dataset', default='USPTO_480k', help='Dataset to use')
    parser.add_argument('-r', '--retro', default=False,  help='Preprcessing for retrosyntheis or forward synthesis (True for retrosnythesis)')
    parser.add_argument('-f', '--force', default=False,  help='Force to preprcess the dataset again')
    parser.add_argument('-v', '--verbose', default=False,  help='Verbose during template extraction')
    parser.add_argument('-stereo', '--use-stereo', default=True,  help='Use stereo info in template extraction')
    parser.add_argument('-symbol', '--use-symbol', default=False,  help='Use atom symbol in template extraction')
    parser.add_argument('-t', '--threshold', type=int, default=1,  help='Template refinement threshold')
    parser.add_argument('-min', '--min-template-n', type=int, default=1,  help='Minimum of template frequency')
    parser.add_argument('-max', '--max-edit-n', type=int, default=13,  help='Maximum number of edit number')
    args = parser.parse_args().__dict__

    template_dicts, template_infos = load_templates(args)
    extractor = build_template_extractor(args)
    test_pre = labeling_dataset(args, 'test', template_dicts, template_infos, extractor)
    val_pre = labeling_dataset(args, 'valid', template_dicts, template_infos, extractor)
    train_pre = labeling_dataset(args, 'train', template_dicts, template_infos, extractor)
    
    combine_preprocessed_data(train_pre, val_pre, test_pre, args)
    
        
        
        