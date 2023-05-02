# Code for Paper: "KGML-xDTD: A Knowledge Graph-based Machine Learning Framework for Drug Treatment Prediction and Mechanism Description"

## Pre-Setting

1. Please install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and install a tool [zenodo-get](https://zenodo.org/record/1261813/) using the command `pip install zenodo-get`

2. Download relevant data and software from Zenodo using the command below:
```Shell
zenodo_get --doi=10.5281/zenodo.7582233
```
Note we provide the associated description of each dataset and software downloaded via this step on [Zenodo](https://zenodo.org/record/7582233). Some datasets are large, so the downloading process needs to take a while.

3. Set up the local neo4j database by the following steps:
```Shell
## assume step 2 has been implemented
tar zxvf neo4j-community-3.5.26.tar.gz
rm neo4j-community-3.5.26.tar.gz
```

4. Please install the relevant conda environments by the following commands:
```Shell
## construct two conda environements
conda env create -f envs/graphsage_p2.7env.yml
conda env create -f envs/main_env.yml

## activiate the 'main_env' conda environment
conda activate main_env

## install pytorch geometric (it might take some time, so you don't need to install it if you don't try to run the baseline models)
TORCH_VERSION=1.10.2 ## check version by executing "python -c 'import torch; print(torch.__version__)'"
CUDA_VERSION='cu113' ## check version by executing "python -c 'import torch; print(torch.version.cuda)'"
pip install torch-scatter==2.0.9 torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-geometric==2.0.3
```
5. Please install OpenKE PyTorch Version by the following commands:
```Shell
## Clone the OpenKE-PyTorch branch
git clone -b OpenKE-PyTorch https://github.com/thunlp/OpenKE --depth 1
mv OpenKE/openke .
rm -rf OpenKE
cd openke
## Compile C++ files
bash make.sh
cd ..
```

6. Build the local Biomedical Medical Knowledge (BKG) by the following commands:
```Shell
## decompress tar.gz file
tar zxvf bkg_rtxkg2c_v2.7.3.tar.gz
rm bkg_rtxkg2c_v2.7.3.tar.gz
ln -s ../bkg_rtxkg2c_v2.7.3/relevant_dbs/node_synonymizer_v1.0_KG2.7.3.sqlite ./scripts/node_synonymizer_v1.0_KG2.7.3.sqlite

## set up neo4j info 
export neo4j_username='neo4j'
export neo4j_password='neo4j' ## if this password doesn't work, please refer to https://neo4j.com/docs/operations-manual/current/configuration/set-initial-password/ to set up a new password
export neo4j_bolt='bolt://localhost:7687'
## For an alternative method, you can copy and paste the above lines into ~/.profile file on your linux machine 

## set up bkg building parameters
database_name='customized_rtxkg2c.db'
neo4j_config='./neo4j-community-3.5.26/conf/neo4j.conf'
path_tsv_folder='./bkg_rtxkg2c_v2.7.3/tsv_files'

## build bkg with neo4j
bash ./bkg_rtxkg2c_v2.7.3/scripts/shell_scripts/read_bkg_to_neo4j.sh ${database_name} ${neo4j_config} ${path_tsv_folder}

## restart neo4j
neo4j_command=`echo ${neo4j_config} | sed 's/conf\/neo4j.conf/bin\/neo4j/'`
${neo4j_command} restart

## add indexes and constraints to the graph database
python ./bkg_rtxkg2c_v2.7.3/scripts/python_scripts/create_indexes_contrains.py
```

7. Apply for DrugBank license and download `drugbank.xml` dataset

Due to the drugbank license limitation, we cannot directly provide the `drugbank.xml` dataset that used in this research. Please first go to [drugbank](https://go.drugbank.com/releases/latest) website, and follow the instruction to access its non-commercial license and then download its dataset `drugbank.xml` containing all drug information.

---

## Data Preprocessing
Pleaase follow the steps 1-4 within `1_data_preprocessing.sh` to do data pre-processing. These steps may need a few hours. Note that the step 3 needs user to get a drugbank academic license to download the `drugbank.xml` file and then put it into the './data' folder.

---

## Model training
Pleaase follow the steps 5-11 within `2_model_training.sh` to do model training. These steps may need a few days. These model training steps include the node-attribute embedding generation (step5) via [PubMedBert](https://arxiv.org/abs/2007.15779) model, [GraphSage](https://arxiv.org/abs/1706.02216) embedding generation (step6-7) via [its official source code](https://github.com/williamleif/GraphSAGE), Random Forest model training, and [Adversarial ActorCritic model](https://www.microsoft.com/en-us/research/uploads/prod/2020/05/sigir_RLRec_camera_ready.pdf) (combined with reward shaping strategy and "demonstration paths"). Note that running GraphSage with its official source code needs to switch to `graphsage_p2.7env` conda environment via command `conda activate graphsage_p2.7env`.

---

## Model Inference
Before doing model inference, plesae make sure the steps 1-6 described in 'Pre-Setting' have been executed and download the trained models from Zenodo via the following commands:
```
cd ./model_evaluation
## download trained model and relevant data
zenodo_get --doi=10.5281/zenodo.7653456
## uncompress files
tar zxvf models.tar.gz
rm models.tar.gz
tar zxvf data.tar.gz
rm data.tar.gz
```

We provide an example below to show how to use KGML-xDTD model framework within `python` environment:

```python
## load packages and KGML_xDTD
import os, sys
path_list = os.getcwd().split('/')
index = path_list.index('KGML-xDTD')
script_path = '/'.join(path_list[:(index+1)] + ['model_evaluation','scripts'])
sys.path.append(script_path)
import argparse
from KGML_xDTD import KGML_xDTD

## set up general parameters
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.use_gpu = True ## set it to False if you don't plan to use GPU
args.gpu = 0

## set up data path and model path
data_path = '/'.join(path_list[:(index+1)] + ['model_evaluation','data'])
model_path = '/'.join(path_list[:(index+1)] + ['model_evaluation','models'])

## create a KGML-xDTD object (this step needs to take around 5 minutes because its need to load the required files (e.g. KG) and trained modules)
xdtd = KGML_xDTD(args, data_path, model_path)

## predict a treatment probability for a single drug-diseae pair
xdtd.predict_single_ddp(drug_name='Eptacog Alfa', disease_name='Hemophilia B')
## predict treatment probabilities for a list of drug-diseae pairs
xdtd.predict_ddps(drug_disease_name_list=[('Eptacog Alfa','Hemophilia B'),('Factor VIIa','Hemophilia B'),('Thrombin','Hemophilia B')])
## predict top 10 potential diseases that could be treated by a given drug
xdtd.predict_top_N_diseases(drug_name='Eptacog Alfa', N=10)
## predict top 10 potential drugs that could be used to treat a given disease
xdtd.predict_top_N_drugs(disease_name='Hemophilia B', N=10)
## predict top 10 potential KG-based MOA paths for explaining the treatment relationship of a single drug-diseae pair
xdtd.predict_top_M_moa_paths(drug_name='Eptacog Alfa', disease_name='Hemophilia B', M=10)
```

### Evaluation
Please run the following command to replicate the model comparison reported in our paper (Table2 and Table3) based on the `test` dataset. You can choose to run a few particular models by specifying them via the parameter `models`. Note that the evaulation of some models (e.g. SVM) are time-consuming.
```Shell
## set working directory
work_folder=$(pwd)

## Evaluation of the drug repurposing predictions (Table2 in paper)
### Note that we don't support GAT model in this script because the total size of its corresponding input files are larger than 300GB, and the public reporistory (e.g. Zenodo) can't host such large data. But we provide all scripts that we used to train GAT under "./baselines/GAT" folder.
python ${work_folder}/model_evaluation/scripts/evaluate_models.py --data_path ${work_folder}/model_evaluation/data \
                                                                  --model_path ${work_folder}/model_evaluation/models \
                                                                  --selected_ddp_models kgml_xdtd_wo_naes kgml_xdtd \
                                                                  --eval_mode 'ddp'

## Evaluation of the KG-based MOA path predictions (Table3 in paper)
python ${work_folder}/model_evaluation/scripts/evaluate_models.py --data_path ${work_folder}/model_evaluation/data \
                                                                  --model_path ${work_folder}/model_evaluation/models \
                                                                  --selected_moa_models "all" \
                                                                  --eval_mode "moa"

```
If you're interested in how to generate some intermediate files for the evaluation above, pleae refer to steps 12-15 in `3_model_evaluation.sh`.

