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
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-geometric
```

5. Build the local Biomedical Medical Knowledge (BKG) by the following commands:
```Shell
## decompress tar.gz file
tar zxvf bkg_rtxkg2c_v2.7.3.tar.gz
rm bkg_rtxkg2c_v2.7.3.tar.gz

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

6. Apply for DrugBank license and download `drugbank.xml` dataset

Due to the drugbank license limitation, we cannot directly provide the `drugbank.xml` dataset that used in this research. Please first go to [drugbank](https://go.drugbank.com/releases/latest) website, and follow the instruction to access its non-commercial license and then download its dataset `drugbank.xml` containing all drug information.


## Data Preprocessing, Model training and Evaluation
Pleaase follow the steps within `main.sh` to do data pre-processing, model training and model evaluation.
