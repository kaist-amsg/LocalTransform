# LocalTransform
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)[![DOI](https://zenodo.org/badge/443246460.svg)](https://zenodo.org/badge/latestdoi/443246460)<br>
Implementation of organic reactivity prediction with LocalTransform developed by Prof. Yousung Jung group at KAIST (contact: ysjn@kaist.ac.kr).<br><br>
![LocalTransform](https://i.imgur.com/9SA50iK.jpg)

## Model size decrease announcement (2022.10.31)
We slightly modified the model architechture to decrease the model size from 59MB to 36.4MB so we can upload to GitHub repo by decrease to size of bond feature from 512 to 256 through bond_net (see `scripts/model.py` for more detail). This modification also accelerate the training process.
Also we fix few part of code to enable smooth implementation on cpu.

## Contents

- [Developer](#developer)
- [OS Requirements](#os-requirements)
- [Python Dependencies](#python-dependencies)
- [Installation Guide](#installation-guide)
- [Reproduce the results](#reproduce-the-results)
- [Demo and human benchmark results](#demo-and-human-benchmark-results)
- [Publication](#publication)
- [License](#license)

## Developer
Shuan Chen (shuankaist@kaist.ac.kr)<br>

## OS Requirements
This repository has been tested on both **Linux** and **Windows** operating systems.

## Python Dependencies
* Python (version >= 3.6) 
* Numpy (version >= 1.16.4) 
* PyTorch (version >= 1.0.0) 
* RDKit (version >= 2019)
* DGL (version >= 0.5.2)
* DGLLife (version >= 0.2.6)

## Installation Guide
Create a virtual environment to run the code of LocalTransform.<br>
Make sure to install pytorch with the cuda version that fits your device.<br>
This process usually takes few munites to complete.<br>
```
git clone https://github.com/kaist-amsg/LocalTransform.git
cd LocalTransform
conda create -c conda-forge -n rdenv  python=3.6 -y
conda activate rdenv
conda install pytorch cudatoolkit=11.3 -c pytorch -y
conda install -c conda-forge rdkit -y
conda install -c dglteam dgl-cuda11.3
pip install dgllife
```

## Reproduce the results
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
The trained model will be saved at `LocalTransform/models/LocalTransform_sep.pth`<br>

### [4] Test LocalTransform model
To use the model to test on test set, simply run 
```
python Test.py -sep True
```
to get the raw prediction file saved at `LocalTransform/outputs/raw_prediction/LocalTransform_sep.txt`<br>
Finally you can get the reactants of each prediciton by decoding the raw prediction file
```
python Decode_predictions.py -sep True
```
The decoded reactants will be saved at 
`LocalTransform/outputs/decoded_prediction/LocalTransform_sep.txt`<br>

### [5] Exact match accuracy calculation
By using
```
python Calculate_topk_accuracy.py -m sep
```
the top-k accuracy will be calculated from the files generated at step [4]

## Demo and human benchmark results
See `Synthesis.ipynb` for running instructions and expected output. Human benchmark results is also shown at the end of the notebook.<br>


## Publication
[A Generalized Template-Based Graph Neural Network for Accurate Organic Reactivity Prediction, Nat Mach Intell 2022](https://www.nature.com/articles/s42256-022-00526-z)

## License
This project is covered under the **Apache 2.0 License**.
