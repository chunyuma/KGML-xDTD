# Code for Paper: "KGML-xDTD: A Knowledge Graph-based Machine Learning Framework for Drug Treatment Prediction and Mechanism Description"

## Set up required running environment

1. Please install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and install a tool [gdown](https://pypi.org/project/gdown/) by using the command `pip install gdown`

1. Please install neo4j database by the following steps:
```Shell
## Assume 'gdown' has been installed. If not, please first follow step 1 to install it.
gdown --id 1UftaTUHQ-NLuhgw51ZY3Ax3Bd6uuHbrS
tar zxvf neo4j-community-3.5.26.tar.gz
cd ..
```

3. Please install the relevant conda environments by the following commands:
```Shell
conda env create -f envs/graphsage_p2.7env.yml
conda env create -f envs/main_env.yml
## activiate 'main_env' conda environment
conda activate main_env
## install pytorch geometric (it might take some time, so you don't need to install it if you don't try to run the baseline models)
TORCH_VERSION=1.10.2 ## check version by executing "python -c 'import torch; print(torch.__version__)'"
CUDA_VERSION='cu113' ## check version by executing "python -c 'import torch; print(torch.version.cuda)'"
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-geometric
```

## Download required dataset
Within this code .zip package, we provide the processed training data collected from four data sources (e.g., MyChem, SemMedDB, NDF-RT and RepoDB) in `training_data.tar.gz`, as well as, the curated MOA paths `indication_paths.yaml` downloaded from [DrugMechDB](https://github.com/SuLab/DrugMechDB).  Please see more data description in the paper. 

However, to run the code, you also need to download some other required datasets by the following steps:

### 1. Our customized knowledge graph based on KG2c
Assume you're working on a Linux system, please use the following command to download the knowledge graph dataset:
```Shell
## download kg from Google Drive (Please install 'gdown' tool via pip first)
gdown --id 1MHQk63GGQ8k58FUGqSO1Z4NPL7M08oSA
tar zxvf bkg_rtxkg2c_v2.7.3.tar.gz
```

### 2. Drugank drug information
Considering the drugbank license problem, we cannot directly provide this dataset. Instead, to access this dataset, you need to first go to [drugbank](https://go.drugbank.com/releases/latest) website, and follow its requirement to access its non-commercial license and then download its dataset `drugbank.xml` containing all drug information.

## Data Preprocessing, model training and evaluation
Pleaase follow the steps within `main.sh` to do data pre-processing, model training and model evaluation.
