import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from rdkit import Chem, RDLogger 
RDLogger.DisableLog('rdApp.*')

# remove the atom map numbers
def demap(smi):
    mol = Chem.MolFromSmiles(smi)
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
    return set(Chem.MolToSmiles(mol).split('.'))

# make sure the product contains more than 1 atom
def clean_product(products):
    products = '.'.join([product for product in products.split('.') if Chem.MolFromSmiles(product).GetNumAtoms() > 1])
    return demap(products)
        
# match if ay least one product is recored in the prediction
def exact_match(reactants, products, preds):
    if len(products) == 0:
        return 1
    try:
        for k, pred in enumerate(preds):
            pred_set = reactants.union(set(pred.split('.')))
            if pred_set.intersection(products):
                return k+1
        return -1
    except:
        return -1
    
def read_ground_truth(args):
    reactants = {}
    products = {}
    with open('../data/%s/test.txt' % args['dataset'], 'r') as f:
        for i, line in enumerate(tqdm(f.readlines(), total=len(f.readlines()), desc='Reading ground truth file...')):
            rxn = line.split(' ')[0]
            reactant, product = rxn.split('>>')
            reactants[i] = demap(reactant)
            products[i] = clean_product(product)
    return reactants, products

def read_predictions(args):
    result_file = '../outputs/decoded_prediction/LocalTransform_%s.txt' % args['model_name']
    results = {}
    scores = {}
    with open(result_file, 'r') as f:
        for i, line in enumerate(tqdm(f.readlines(), total=len(f.readlines()), desc='Reading prediction file...')):
            line = line.split('\n')[0]
            i = int(line.split('\t')[0])
            predictions = line.split('\t')[1:]
            results[i] = [p.split(', ')[0] for p in predictions]
            scores[i] = [p.split(', ')[1] for p in predictions]
    return results, scores

def main(args):
    reactants, products = read_ground_truth(args)
    results, scores = read_predictions(args)
    exact_matches = []
    for i in range(len(results)):
        exact_matches.append(exact_match(reactants[i], products[i], results[i]))
        if i % 100 == 0:
            print ('\rCalculating top-k exact accuracy... %s/%s' % (i, len(results)), end='', flush=True)
    print ()
    ks = [1, 2, 3, 5]
    exact_accu = {k:0 for k in ks}
    for i in range(len(exact_matches)):
        for k in ks:
            if exact_matches[i] <= k and exact_matches[i] != -1:
                exact_accu [k] += 1
    for k in ks:
        print ('Top-%d accuracy: %.3f,' % (k, exact_accu[k]/len(results)))
    
if __name__ == '__main__':
    parser = ArgumentParser('Training arguements')
    parser.add_argument('-d', '--dataset', default='USPTO_480k', help='Dataset to use')
    parser.add_argument('-m', '--model_name', type=str, default='mix', help='Model name')
    args = parser.parse_args().__dict__
    main(args)