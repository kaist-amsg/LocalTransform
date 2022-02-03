import os, sys, re
import pandas as pd
import multiprocessing
from tqdm import tqdm
from functools import partial
from collections import defaultdict
from argparse import ArgumentParser
sys.path.append('../')
    
import rdkit
from rdkit import Chem, RDLogger 
from rdkit.Chem import rdChemReactions

from utils import *
from dataset import combine_reactants
from LocalTemplate.template_collector import Collector

def get_k_predictions(test_id, args):
    raw_predictions = args['raw_predictions'][test_id]
    reactants = raw_prediction[0]
    reagents = raw_prediction[1]
    if not args['sep']:
        reactants, reagents = combine_reactants(reactants, reagents)
    collector = Collector(reactants, args['template_infos'], reagents, intermolecule = args['sep'], verbose > 1)
    predictions = raw_prediction[2:]
    for prediction in predictions:
        pred_type, pred_site, pred_template_class, pred_score = eval(prediction) # (edit_type, pred_site, pred_template_class, pred_score)
        template, H_code, C_code, action = args['template_dicts'][pred_type][pred_template_class]
        collector.collect(template, H_code, C_code, action, pred_site, pred_score)
        if len(collector.predictions) >= args['top_k']:
            break
    decoded_predictions = [(k, v['score']) for k, v in sorted(collector.predictions.items(), key=lambda item: -item[1]['score'])]
    
    return test_id, decoded_predictions

def main(args):   
    template_dicts = {}
    for site in ['real', 'virtual']:
        template_df = pd.read_csv('%s/%s_templates.csv' % (args['data_dir'], site))
        template_dicts[site[0]] = {template_df['Class'][i]: template_df['Template'][i].split('_') for i in template_df.index}
    template_df = pd.read_csv('%s/template_infos.csv' % args['data_dir'])
    
    args['template_dicts'] = template_dicts
    args['template_infos'] = {template_infos['Template'][i]: {'edit_site': eval(template_df['edit_site'][i]), 'change_H': eval(template_df['change_H'][i]), 'change_C': eval(template_df['change_C'][i])} for i in template_df.index}
    
    if args['model_name'] == 'default':
        if args['sep']:
            result_name = 'LocalTransform_sep.txt'
        else:
            result_name = 'LocalTransform_mix.txt'
    else:
        result_name = 'LocalTransform_%s.txt' % args['model_name']
        
    prediction_file =  '../outputs/raw_prediction/' + result_name
    raw_predictions = {}
    with open(prediction_file, 'r') as f:
        for line in f.readlines():
            seps = line.split('\t')
            if seps[0] == 'Test_id':
                continue
            raw_predictions[int(seps[0])] = seps[1:]
        
    output_path = '../outputs/decoded_prediction/' + result_name
    args['raw_predictions'] = raw_predictions
    
    # decode the result with multi-processing
    result_dict = {}
    partial_func = partial(get_k_predictions, args = args)
    with multiprocessing.Pool(processes=8) as pool:
        tasks = range(len(raw_predictions))
        for result in tqdm(pool.imap_unordered(partial_func, tasks), total=len(tasks), desc='Decoding LocalTransform predictions'):
            result_dict[result[0]] = result[1]
    
        
    with open(output_path, 'w') as f:
        for i in sorted(result_dict.keys()) :
            predictions = result_dict[i]
            f.write('\t'.join([str(i)] + predictions) + '\n')
            print('\rWriting LocalTransform predictions %d/%d' % (i, len(raw_predictions)), end='', flush=True)
    print ()
       
if __name__ == '__main__':      
    parser = ArgumentParser('Decode Prediction')
    parser.add_argument('-d', '--dataset', default='USPTO_480k', help='Dataset to use')
    parser.add_argument('-m', '--model-name', default='default', help='Model to use')
    parser.add_argument('-s', '--sep', default=False, help='Train the model with reagent seperated or not')
    parser.add_argument('-k', '--top-k', default= 5, help='Number of top predictions')
    args = parser.parse_args().__dict__
    mkdir_p('../outputs/decoded_prediction')
    main(args) 
    