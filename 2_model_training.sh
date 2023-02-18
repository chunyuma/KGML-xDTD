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

## step5: generate node-attribute embedding via PubMedBert model
python ${work_folder}/scripts/calculate_attribute_embedding.py --log_dir ${work_folder}/log_folder \
                                                               --log_name step5.log \
                                                               --data_dir ${work_folder}/data \
                                                               --use_gpu \
                                                               --gpu ${gpu} \
                                                               --output_folder ${work_folder}/data


# step6: generate graphsage input files
echo 'running step6: generate graphsage input files'
python ${work_folder}/scripts/graphsage_data_generation.py --log_dir ${work_folder}/log_folder \
                                                           --log_name step6_1.log \
                                                           --data_dir ${work_folder}/data \
                                                           --emb_file ${work_folder}/data/text_embedding/embedding_biobert_namecat.pkl \
                                                           --process 80 \ ## change it based on your computer CPU cores
                                                           --validation_percent 0.3 \
                                                           --output_folder ${work_folder}/data/graphsage_input

python ${work_folder}/scripts/generate_random_walk.py --log_dir ${work_folder}/log_folder \
                                                      --log_name step6_2.log \
                                                      --Gjson ${work_folder}/data/graphsage_input/data-G.json \
                                                      --walk_length 100 \
                                                      --number_of_walks 10 \
                                                      --batch_size 200000 \
                                                      --process 200 \ ## change it based on your computer CPU cores
                                                      --output_folder ${work_folder}/data/graphsage_input

# ## step7: generate graphsage embedding
echo 'running step7: generate graphsage embedding'
## set graphsage folder
## in the original source code of GraphSAGE (if you download from https://github.com/williamleif/GraphSAGE), remember to set the parameter 'normalize' of load_data function to false and 'load_walks' to True
## set python path (Please use python 2.7 to run graphsage as graphsage was written by python2.7)
ppath=~/anaconda3/envs/graphsage_p2.7env/bin/python ## please change this path to where the python located in "graphsage_p2.7env" conda environment
ln -s ${work_folder}/scripts/GraphSAGE/graphsage

## run graphsage unsupervised model
${ppath} -m graphsage.unsupervised_train --train_prefix ${work_folder}/data/graphsage_input/data \
                                         --model_size 'big' \
                                         --learning_rate 0.001 \
                                         --epochs 10 \
                                         --samples_1 96 \
                                         --samples_2 96 \
                                         --dim_1 256 \
                                         --dim_2 256 \
                                         --model 'graphsage_mean' \
                                         --max_total_steps 10000 \
                                         --validate_iter 100 \
                                         --batch_size 512 \
                                         --max_degree 96

## step7: transform the GraphSage results to .emb format
echo 'running step7: transform the GraphSage results to .emb format'
python ${work_folder}/scripts/transform_format.py --log_dir ${work_folder}/log_folder \
                                                  --log_name step7.log \
                                                  --data_dir ${work_folder}/data \
                                                  --input ${work_folder}/unsup-graphsage_input/graphsage_mean_big_0.001000

## step8: pretrain RF model
echo 'running step8: pretrain RF model'
python ${work_folder}/scripts/run_RF_model_2class.py --log_dir ${work_folder}/log_folder \
                                                     --log_name step8_rf_2class.log \
                                                     --data_dir ${work_folder}/data \
                                                     --pair_emb ${pair_emb_method} \
                                                     --output_folder ${work_folder}/models

python ${work_folder}/scripts/run_RF_model_3class.py --log_dir ${work_folder}/log_folder \
                                                     --log_name step8_rf_3class.log \
                                                     --data_dir ${work_folder}/data \
                                                     --pair_emb ${pair_emb_method} \
                                                     --output_folder ${work_folder}/models

python ${work_folder}/scripts/run_logistic_model_2class.py --log_dir ${work_folder}/log_folder \
                                                           --log_name step8_logistic_2class.log \
                                                           --data_dir ${work_folder}/data \
                                                           --pair_emb ${pair_emb_method} \
                                                           --output_folder ${work_folder}/models

python ${work_folder}/scripts/run_svm_model_2class.py --log_dir ${work_folder}/log_folder \
                                                      --log_name step8_svm_2class.log \
                                                      --data_dir ${work_folder}/data \
                                                      --pair_emb ${pair_emb_method} \
                                                      --output_folder ${work_folder}/models

# step9: generate pre-calculated transition file for demonstration paths
echo 'running step9: generate precalculated transition file for demonstration paths'
python ${work_folder}/scripts/generate_expert_trans.py --log_dir ${work_folder}/log_folder \
                                                       --log_name step9.log \
                                                       --data_dir ${work_folder}/data \
                                                       --path_file_name 'train_expert_demonstration_relation_entity_max'${max_path}'_filtered.pkl' \
                                                       --max_path ${max_path} \
                                                       --state_history ${state_history} \
                                                       --expert_trains_file_name 'train_expert_transitions_history'${state_history}'.pkl'

# step10: pretrained ActorCritic model by the behavior cloning
echo 'running step10: pretrained ActorCritic model by the behavior cloning'
python ${work_folder}/scripts/run_pretrain_ac_model.py --log_dir ${work_folder}/log_folder \
                                                       --log_name step10.log \
                                                       --data_dir ${work_folder}/data \
                                                       --path_file_name 'train_expert_demonstration_relation_entity_max'${max_path}'_filtered.pkl' \
                                                       --text_emb_file_name 'embedding_biobert_namecat.pkl' \
                                                       --output_folder ${work_folder}/models \
                                                       --max_path ${max_path} \
                                                       --bandwidth ${max_neighbor} \
                                                       --bucket_interval ${bucket_interval} \
                                                       --pretrain_model_path ${work_folder}/models/RF_model_3class/RF_model.pt \
                                                       --use_gpu \
                                                       --gpu ${gpu} \
                                                       --batch_size ${pre_batch_size} \
                                                       --max_pre_path ${max_pre_path} \
                                                       --epochs 20 \
                                                       --pre_actor_epoch 10 \
                                                       --hidden 512 512 \
                                                       --state_history ${state_history} \
                                                       --lr ${learning_rate} \
                                                       --scheduler_patience 5 \
                                                       --scheduler_factor 0.1

# step11: train Adversarial ActorCritic model
echo 'running step11: train Adversarial ActorCritic model'
python ${work_folder}/scripts/run_adac_model.py --log_dir ${work_folder}/log_folder \
                                                --log_name step11.log \
                                                --data_dir ${work_folder}/data \
                                                --path_file_name 'train_expert_demonstration_relation_entity_max'${max_path}'_filtered.pkl' \
                                                --text_emb_file_name 'embedding_biobert_namecat.pkl' \
                                                --path_trans_file_name 'train_expert_transitions_history2.pkl' \
                                                --output_folder ${work_folder}/models \
                                                --max_path ${max_path} \
                                                --bandwidth ${max_neighbor} \
                                                --bucket_interval ${bucket_interval} \
                                                --pretrain_model_path ${work_folder}/models/RF_model_3class/RF_model.pt \
                                                --use_gpu --gpu ${gpu} --train_batch_size ${train_batch_size} \
                                                --warmup --pre_ac_file 'pre_model_epoch20.pt' \
                                                --disc_hidden 512 512 \
                                                --metadisc_hidden 512 256 \
                                                --ac_hidden 512 512 \
                                                --epochs ${num_epochs} \
                                                --state_history ${state_history} \
                                                --ac_update_delay 50 \
                                                --ent_weight ${entropy_weight} \
                                                --disc_alpha 0.006 \
                                                --metadisc_alpha 0.012 \
                                                --num_rollouts ${num_rollouts} \
                                                --act_dropout ${action_dropout} \
                                                --ac_lr ${learning_rate} \
                                                --disc_lr ${learning_rate} \
                                                --metadisc_lr ${learning_rate}


# Note that some baseline models are impletmented independently, please go to ./baselines folder to find the specific baseline models that we compare in our paper. Under each baseline model folder, there is a "main.sh" file. Please follow the steps within "main.sh" file to implement such baseline model.