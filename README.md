# LocalTransform
Implementation of Reaction outcome prediction with LocalTransform developed by prof. Yousung Jung group at KAIST (contact: ysjn@kaist.ac.kr).

## Developer
Shuan Chen (shuankaist@kaist.ac.kr)<br>

## Requirements
* Python (version >= 3.6) 
* Numpy (version >= 1.16.4) 
* PyTorch (version >= 1.0.0) 
* RDKit (version >= 2019)
* DGL (version >= 0.5.2)
* DGLLife (version >= 0.2.6)

## Requirements
Create a virtual environment to run the code of LocalTransform.<br>
Install pytorch with the cuda version that fits your device.<br>
```
cd LocalTransform
conda create -c conda-forge -n rdenv  python=3.6 -y
conda activate rdenv
conda install pytorch==1.0.0 cuda80 -c pytorch -y
conda install -c conda-forge rdkit -y
pip install dgl
pip install dgllife
```

## Publication
Prediction of Organic Reaction Outcomes by Chemist-Like Machine Intelligence and Generalized Reaction Templates (under review).


## Usage
### [1] Download the raw data of USPTO-480k dataset
Download the data from https://github.com/wengong-jin/nips17-rexgen/blob/master/USPTO/ and move the data to `./data/USPTO_480k/`.

### [2] Data preprocessing
A two-step data preprocessing is needed to train the LocalTransform model.

#### 1) Local reaction template derivation 
First go to the data processing folder
```
cd preprocessing
```
and extract the reaction templates.
```
python Extract_from_train_data.py
```
This will give you four files, including 
(1) real_templates.csv (reaction templates for real bonds)
(2) virtual_templates.csv (reaction templates for imaginary bonds)
(3) template_infos.csv (including the hydrogen change, charge change and action information)<br>

#### 2) Assign the derived templates to raw data
By running
```
python Run_preprocessing.py
```
You can get four preprocessed files, including 
(1) preprocessed_train.csv
(2) preprocessed_valid.csv
(3) preprocessed_test.csv
(4) labeled_data.csv<br>


### [3] Train LocalTransform model
Go to the main scripts folder
```
cd ../scripts
```
and run the following to train the model with reagent seperated or not (default: False)
```
python Train.py -sep True
```
The trained model will be saved at ` LocalTransform/models/LocalTransform_sep.pth`<br>

### [4] Test LocalTransform model
To use the model to test on test set, simply run 
```
python Test.py -sep True
```
to get the raw prediction file saved at ` LocalTransform/outputs/raw_prediction/LocalTransform_sep.txt`<br>
Finally you can get the reactants of each prediciton by decoding the raw prediction file
```
python Decode_predictions.py -sep True
```
The decoded reactants will be saved at 
`LocalTransform/outputs/decoded_prediction/LocalTransform_sep.txt`<br>


## Human benchmark results and implementation on Jupyter Notebook
For human benchmark results and quick implementation of organic reaction outcome prediction via LocalTransform, see `Synthesis.ipynb` for quick start.

#### Exact match accuracy (%) on USPTO-480k dataset at seperated prediction scenario

| Method | Top-1 | Top-2 | Top-3 | Top-5 |
| -------- | -------- | -------- | -------- | -------- |
| Molecular Transformer | 90.4 | 93.7 | 94.6 | 95.3 |
| Augmented Transformer | 91.9 | 95.4 | / | 97.0 |
| WLDN         | 85.6 | 90.5 | 92.8 | 93.4 |
| MEGAN        | 89.3 | 92.7 | 94.4 | 95.6 |
| Symbolic  | 90.4 | 93.2 | 94.1 | 95.0 |
| NERF     | 90.7 | 92.3 | 93.3 | 93.7 |
| LocalTransform  | **92.3** | **95.6** | **96.5** | **97.2** |

#### Exact match accuracy (%) on USPTO-480k dataset at mixed prediction scenario

| Method | Top-1 | Top-2 | Top-3 | Top-5 |
| -------- | -------- | -------- | -------- | -------- |
| Molecular Transformer | 88.7 | 92.1 | 93.1 | 94.2 |
| Augmented Transformer | 90.6 | 94.4 | / | 96.1 |
| Graph2SMILES    | 90.3 | / | 94.0 | 94.8 |
| MEGAN        | 86.3 | 90.3 | 92.4 | 94.0 |
| LocalTransform  | **90.8** | **94.8** | **95.7** | **96.3** |


