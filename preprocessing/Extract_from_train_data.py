from collections import defaultdict
import pandas as pd
import sys, os, re
import rdkit
from rdkit import Chem, RDLogger 
from rdkit.Chem import rdChemReactions
RDLogger.DisableLog('rdApp.*')
sys.path.append('../')
from LocalTemplate.template_extractor import extract_from_reaction

def build_template_extractor(args):
    setting = {'verbose': False, 'use_stereo': False, 'use_symbol': False, 'max_unmap': 5, 'retro': False, 'remote': True, 'least_atom_num': 2}
    for k in setting.keys():
        if k in args.keys():
            setting[k] = args[k]
    print ('Template extractor setting:', setting)
    return lambda x: extract_from_reaction(x, setting)

def get_reaction_template(extractor, rxn, _id = 0):
    rxn = {'reactants': rxn.split('>>')[0], 'products': rxn.split('>>')[1], '_id': _id}
    result = extractor(rxn)
    return rxn, result

def get_full_template(template, H_change, Charge_change, Chiral_change):
    H_code = ''.join([str(H_change[k+1]) for k in range(len(H_change))])
    Charge_code = ''.join([str(Charge_change[k+1]) for k in range(len(Charge_change))])
    Chiral_code = ''.join([str(Chiral_change[k+1]) for k in range(len(Chiral_change))])
    return '_'.join([template, H_code, Charge_code, Chiral_code])

def extract_templates(args, extractor):
    rxns = {}
    with open('../data/%s/train.txt' % args['dataset'], 'r') as f:
        for i, line in enumerate(f.readlines()):
            rxns[i] = line.split(' ')[0]
    
    TemplateEdits = {}
    TemplateCs = {}
    TemplateHs = {}
    TemplateSs = {}
    TemplateFreq = defaultdict(int)
    real_templates = defaultdict(int)
    virtual_templates = defaultdict(int)
    
    for i, reaction in rxns.items():
        try:
            rxn, result = get_reaction_template(extractor, reaction, i)
            if 'reactants' not in result:
                print ('\ntemplate problem: id: %s' % i)
                continue
            reactant = result['reactants']
            if 'reaction_smarts' not in result.keys():
                continue
            template = result['reaction_smarts']
            edits = result['edits']
            H_change, Charge_change, Chiral_change = result['H_change'], result['Charge_change'], result['Chiral_change']
            template = get_full_template(template, H_change, Charge_change, Chiral_change)

            if template not in TemplateEdits.keys(): # first come first serve
                TemplateEdits[template] = {edit_type: edits[edit_type][2] for edit_type in edits}
                TemplateHs[template] = H_change
                TemplateCs[template] = Charge_change
                TemplateSs[template] = Chiral_change
                
            TemplateFreq[template] += 1

            for edit_type in edits:
                bonds = edits[edit_type][0]
                if len(bonds) > 0:
                    if edit_type != 'A':
                        real_templates['%s_%s' % (template, edit_type)] += 1
                    else:
                        virtual_templates['%s_%s' % (template, edit_type)] += 1
                
        except KeyboardInterrupt:
            print('Interrupted')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
        except Exception as e:
            print ('\nexception occur! id: %s' % i)
            print (e)
            print (reaction)
            
        if i % 100 == 0:
            print ('\r i = %s, # of template: %s, # of real template: %s, # of virtual template: %s' % (i, len(TemplateFreq), len(real_templates), len(virtual_templates)), end='', flush=True)
    print ('\n total # of template: %s' %  len(TemplateFreq))
    derived_templates = {'real':real_templates, 'virtual': virtual_templates}
        
    TemplateInfos = pd.DataFrame({'Template': k, 'edit_site':TemplateEdits[k], 'change_H': TemplateHs[k], 'change_C': TemplateCs[k], 'change_S': TemplateSs[k], 'Frequency': TemplateFreq[k]} for k in TemplateHs.keys())
    TemplateInfos.to_csv('../data/%s/template_infos.csv' % args['dataset'])
    
    return derived_templates

def export_template(derived_templates, args):
    for k in derived_templates.keys():
        local_templates = derived_templates[k]
        templates = []
        template_class = []
        template_freq = []
        sorted_tuples = sorted(local_templates.items(), key=lambda item: item[1])
        c = 1
        for t in sorted_tuples:
            templates.append(t[0])
            template_freq.append(t[1])
            template_class.append(c)
            c += 1
        template_dict = {templates[i]:i+1  for i in range(len(templates)) }
        template_df = pd.DataFrame({'Template' : templates, 'Frequency' : template_freq, 'Class': template_class})

        template_df.to_csv('../data/%s/%s_templates.csv' % (args['dataset'], k))
    return


if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser('Local Template Extractor')
    parser.add_argument('-d', '--dataset', default='USPTO_480k', help='Dataset to use')
    parser.add_argument('-r', '--retro', default=False,  help='Retrosyntheis or forward synthesis (True for retrosnythesis)')
    parser.add_argument('-v', '--verbose', default=False,  help='Verbose during template extraction')
    parser.add_argument('-m', '--max-edit-n', default=8,  help='Maximum number of edit number')
    parser.add_argument('-stereo', '--use-stereo', default=True,  help='Use stereo info in template extraction')
    parser.add_argument('-symbol', '--use-symbol', default=False,  help='Use atom symbol in template extraction')
    args = parser.parse_args().__dict__
    extractor = build_template_extractor(args)
    derived_templates = extract_templates(args, extractor)
    export_template(derived_templates, args)

    