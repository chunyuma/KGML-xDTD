#!/usr/bin/env bash

## set working directory
work_folder=$(pwd)

############## set up global parameters ##############
## set up neo4j info 
export neo4j_username='neo4j'
export neo4j_password='neo4j' ## if this password doesn't work, please refer to https://neo4j.com/docs/operations-manual/current/configuration/set-initial-password/ to set up a new password
export neo4j_bolt='bolt://localhost:7687'
## For an alternative method, you can copy and paste the above lines into ~/.profile file on your linux machine

## set up hyperparameters for drug repurposing model training
pair_emb_method='concatenate'

## set up hyperparameters for ADAC-based RL model training
max_path=3
state_history=2
max_neighbor=3000
bucket_interval=50
gpu=0
pre_batch_size=1024 
max_pre_path=10000000
num_epochs=100
entropy_weight=0.005
learning_rate=0.0005
action_dropout=0.5
num_rollouts=35
train_batch_size=1120
eval_batch_size=5
factor=0.9

######################################################

## create required folders
if [ ! -d "${work_folder}/data" ]
then:
	mkdir ${work_folder}/data
fi
if [ ! -d "${work_folder}/log_folder" ]
then
	mkdir ${work_folder}/log_folder
fi
if [ ! -d "${work_folder}/models" ]
then
	mkdir ${work_folder}/models
fi
if [ ! -d "${work_folder}/results" ]
then
	mkdir ${work_folder}/results
fi

## move training data to data folder
if [ ! -d "${work_folder}/data/training_data" ]
then
    mv ${work_folder}/training_data.tar.gz ${work_folder}/data
    cd ${work_folder}/data
    tar zxvf training_data.tar.gz
    rm training_data.tar.gz
    cd ${work_folder}
fi

if [ ! -f "${work_folder}/data/indication_paths.yaml" ]
then
    mv ${work_folder}/indication_paths.yaml ${work_folder}/data
fi

## set up node synonymizer
if [ ! -f "${work_folder}/scripts/node_synonymizer_v1.0_KG2.7.3.sqlite" ]
then
    cd ${work_folder}/scripts
    ln -s ${work_folder}/bkg_rtxkg2c_v2.7.3/relevant_dbs/node_synonymizer_v1.0_KG2.7.3.sqlite
    cd ${work_folder}
fi

## step1: download graph from neo4j database
echo "running step1: download graph from neo4j database"
python ${work_folder}/scripts/download_data_from_neo4j.py --log_dir ${work_folder}/log_folder \ 
                                                          --log_name step1.log \
                                                          --output_folder ${work_folder}/data

## step2: generate tp and tn
echo 'running step2: generate tp and tn edges'
python ${work_folder}/scripts/generate_tp_tn_pairs.py --log_dir ${work_folder}/log_folder \
                                                      --log_name step2.log \ 
                                                      --use_input_training_edges \
                                                      --graph ${work_folder}/data/graph_edges.txt \
                                                      --tncutoff "10" \
                                                      --tpcutoff "10" \
                                                      --ngdcutoff "0.6" \
                                                      --tp ${work_folder}/data/training_data/mychem_tp.txt ${work_folder}/data/training_data/semmed_tp.txt ${work_folder}/data/training_data/ndf_tp.txt ${work_folder}/data/training_data/repoDB_tp.txt \
                                                      --tn ${work_folder}/data/training_data/mychem_tn.txt ${work_folder}/data/training_data/semmed_tn.txt ${work_folder}/data/training_data/ndf_tn.txt ${work_folder}/data/training_data/repoDB_tn.txt \
                                                      --output_folder ${work_folder}/data

## step3: preprocess data
echo "running step3: preprocess data"
python ${work_folder}/scripts/preprocess_data.py --log_dir ${work_folder}/log_folder \
                                                 --log_name step3_1.log \
                                                 --data_dir ${work_folder}/data \
                                                 --output_folder ${work_folder}/data

python ${work_folder}/scripts/process_drugbank_action_desc.py --log_dir ${work_folder}/log_folder \
                                                              --log_name step3_2.log \
                                                              --data_dir ${work_folder}/data ## this step needs to request a drugbank academic license to download the drugbank.xml file and then put it into the '${work_folder}/data' folder

python ${work_folder}/scripts/integrate_drugbank_and_molepro_data.py --log_dir ${work_folder}/log_folder \
                                                                     --log_name step3_3.log \
                                                                     --data_dir ${work_folder}/data

python ${work_folder}/scripts/check_reachable.py --log_dir ${work_folder}/log_folder \
                                                 --log_name step3_4.log \
                                                 --data_dir ${work_folder}/data \
                                                 --max_path ${max_path} \
                                                 --bandwidth ${max_neighbor}

python ${work_folder}/scripts/generate_expert_paths.py --log_dir ${work_folder}/log_folder \
                                                       --log_name step3_5.log \
                                                       --data_dir ${work_folder}/data \
                                                       --max_path ${max_path} \
                                                       --bandwidth ${max_neighbor} \
                                                       --ngd_db_path ${work_folder}/bkg_rtxkg2c_v2.7.3/relevant_dbs/curie_to_pmids_v1.0_KG2.7.3.sqlite

## step4: generate the 'treat' and 'not treat' train, val and test dataset
echo "running step4: generate the 'treat' and 'not treat' train, val and test data set"
python ${work_folder}/scripts/split_data_train_val_test.py --log_dir ${work_folder}/log_folder \
                                                           --log_name step4.log \
                                                           --data_dir ${work_folder}/data \
                                                           --n_random_test 500 \
                                                           --n_random 30 --train_val_test_size "[0.8, 0.1, 0.1]" 

