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
    r, v = 'r', 'v'
    raw_prediction = args['raw_predictions'][test_id]
    reactants = raw_prediction[0]
    reagents = raw_prediction[1]
    if not args['sep']:
        reactants = combine_reactants(reactants, reagents)
    collector = Collector(reactants, args['template_infos'], reagents, None, args['sep'], 0)
    predictions = raw_prediction[2:]
    tos = time.time()
    for prediction in predictions:
        pred_type, pred_site, pred_template_class, pred_score = eval(prediction) # (edit_type, pred_site, pred_template_class, pred_score)
        template, H_code, C_code, S_code, action = args['template_dicts'][pred_type][pred_template_class]
        collector.collect(template, H_code, C_code, S_code, action, pred_site, pred_score)
        if len(collector.predictions) >= args['top_k'] or time.time() - tos > 10:
            break
    decoded_predictions = ['%s, %.3f' % (k, v['score']) for k, v in sorted(collector.predictions.items(), key=lambda item: -item[1]['score'])]
    
    return decoded_predictions

def main(args):   
    template_dicts = {}
    for site in ['real', 'virtual']:
        template_df = pd.read_csv('../data/%s/%s_templates.csv' % (args['dataset'], site))
        template_dicts[site[0]] = {template_df['Class'][i]: template_df['Template'][i].split('_') for i in template_df.index}
    template_df = pd.read_csv('../data/%s/template_infos.csv' % args['dataset'])
    
    args['template_dicts'] = template_dicts
    args['template_infos'] = {template_df['Template'][i]: {'edit_site': eval(template_df['edit_site'][i]), 'change_H': eval(template_df['change_H'][i]), 'change_C': eval(template_df['change_C'][i]), 'change_S': eval(template_df['change_S'][i])} for i in template_df.index}
    
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
    
    # decode the result
    with open(output_path, 'w') as f:
        for i in tqdm(range(len(raw_predictions)), total=len(raw_predictions), desc='Decoding LocalTransform predictions'):
            decoded_predictions = get_k_predictions(i, args)
            f.write('\t'.join([str(i)] + decoded_predictions) + '\n')

if __name__ == '__main__':      
    parser = ArgumentParser('Decode Prediction')
    parser.add_argument('-d', '--dataset', default='USPTO_480k', help='Dataset to use')
    parser.add_argument('-m', '--model-name', default='default', help='Model to use')
    parser.add_argument('-s', '--sep', default=False, help='Train the model with reagent seperated or not')
    parser.add_argument('-k', '--top-k', default= 5, help='Number of top predictions')
    args = parser.parse_args().__dict__
    mkdir_p('../outputs/decoded_prediction')
    main(args) 
    