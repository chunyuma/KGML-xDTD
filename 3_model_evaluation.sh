#!/usr/bin/env bash

## set working directory
work_folder=$(pwd)

############## Please set up the following parameters first ##############
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

##########################################################################

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


# step12: evaluate Adversarial ActorCritic model
python ${work_folder}/scripts/evaluate_model.py --log_dir ${work_folder}/log_folder --log_name step12.log --data_dir ${work_folder}/data --policy_net_folder ${work_folder}/models/ADAC_model/policy_net --output_folder ${work_folder} --max_path ${max_path} --bandwidth ${max_neighbor} --bucket_interval ${bucket_interval} --pretrain_model_path ${work_folder}/models/RF_model_3class/RF_model.pt --use_gpu --gpu ${gpu} --eval_batch_size ${eval_batch_size} --topk 50 --state_history ${state_history} --ac_hidden 512 512 --disc_hidden 512 512 --metadisc_hidden 512 256 --factor ${factor}

# Note that some baseline models are impletmented independently, please go to ${work_folder}/baselines folder to find the specific baseline models that we compare in our paper. Under each baseline model folder, there is a "main.sh" file. Please follow the steps within "main.sh" file to implement such baseline model.
# step13: evaluate the predicted paths
## process drugmechdb data
python ${work_folder}/scripts/process_drugmechdb_data.py --log_dir ${work_folder}/log_folder --log_name step13_1.log --data_dir ${work_folder}/data --drugmechDB_yaml ${work_folder}/data/indication_paths.yaml --output_folder ${work_folder}/drugmeshdb
python ${work_folder}/scripts/calculate_path_prob.py --log_dir ${work_folder}/log_folder --log_name step13_2.log --data_dir ${work_folder}/data --policy_net_file ${work_folder}/models/ADAC_model/policy_net/policy_model_epoch51.pt --target_paths_file ${work_folder}/drugmeshdb/res_all_paths.pkl --output_file ${work_folder}/drugmeshdb/res_all_paths_prob.pkl --max_path ${max_path} --bandwidth ${max_neighbor} --bucket_interval ${bucket_interval} --pretrain_model_path ${work_folder}/models/RF_model_3class/RF_model.pt --use_gpu --gpu ${gpu} --state_history ${state_history} --ac_hidden 512 512 --disc_hidden 512 512 --metadisc_hidden 512 256 --batch_size 1000 --factor ${factor} 
# python ${work_folder}/scripts/calculate_evaluation_metrics.py --log_dir ${work_folder}/log_folder --log_name step13_3.log --data_dir ${work_folder}/data --drugmeshdb_match ${work_folder}/drugmeshdb/match_paths.pkl --all_paths_prob ${work_folder}/drugmeshdb/res_all_paths_prob.pkl

# step14: ablation study: train Adversarial ActorCritic model without demonstration paths
echo 'running step14: train Adversarial ActorCritic model without demonstration paths'
python ${work_folder}/scripts/run_adac_model_ablation.py --log_dir ${work_folder}/log_folder_bk --log_name step14.log --data_dir ${work_folder}/data --path_file_name 'train_expert_demonstration_relation_entity_max'${max_path}'_filtered.pkl' --text_emb_file_name 'embedding_biobert_namecat.pkl' --path_trans_file_name 'train_expert_transitions_history2.pkl' --output_folder ${work_folder}/models --max_path ${max_path} --bandwidth ${max_neighbor} --bucket_interval ${bucket_interval} --pretrain_model_path ${work_folder}/models/RF_model_3class/RF_model.pt --use_gpu --gpu 3 --train_batch_size ${train_batch_size} --disc_hidden 512 512 --metadisc_hidden 512 256 --ac_hidden 512 512 --epochs 100 --state_history 2 --ac_update_delay 0 --ent_weight ${entropy_weight} --disc_alpha 0 --metadisc_alpha 0 --num_rollouts ${num_rollouts} --act_dropout ${action_dropout}

# step15: evaluate Adversarial ActorCritic model without demonstration paths
echo 'running step15: evaluate Adversarial ActorCritic model without demonstration paths'
python ${work_folder}/scripts/evaluate_model.py --log_dir ${work_folder}/log_folder --log_name step15_1.log --data_dir ${work_folder}/data --policy_net_folder ${work_folder}/models/ADAC_model_ablation/policy_net --output_folder ${work_folder} --max_path ${max_path} --bandwidth ${max_neighbor} --bucket_interval ${bucket_interval} --pretrain_model_path ${work_folder}/models/RF_model_3class/RF_model.pt --use_gpu --gpu ${gpu} --eval_batch_size ${eval_batch_size} --topk 50 --state_history ${state_history} --ac_hidden 512 512 --disc_hidden 512 512 --metadisc_hidden 512 256 --factor ${factor}
python ${work_folder}/scripts/calculate_path_prob.py --log_dir ${work_folder}/log_folder --log_name step15_2.log --data_dir ${work_folder}/data --policy_net_file ${work_folder}/models/ADAC_model_ablation/policy_net/policy_model_epoch98.pt --target_paths_file ${work_folder}/drugmeshdb/res_all_paths.pkl --output_file ${work_folder}/drugmeshdb/res_all_paths_prob_ablation.pkl --max_path ${max_path} --bandwidth ${max_neighbor} --bucket_interval ${bucket_interval} --pretrain_model_path ${work_folder}/models/RF_model_3class/RF_model.pt --use_gpu --gpu ${gpu} --state_history ${state_history} --ac_hidden 512 512 --disc_hidden 512 512 --metadisc_hidden 512 256 --batch_size 1000 --factor ${factor}
# python ${work_folder}/scripts/calculate_evaluation_metrics.py --log_dir ${work_folder}/log_folder --log_name step15_3.log --data_dir ${work_folder}/data --drugmeshdb_match ${work_folder}/drugmeshdb/match_paths.pkl --all_paths_prob ${work_folder}/drugmeshdb/res_all_paths_prob_ablation.pkl

# Note that some baseline models are impletmented independently, please go to ${work_folder}/baselines folder to find the specific baseline models that we compare in our paper. Under each baseline model folder, there is a "main.sh" file. Please follow the steps within "main.sh" file to implement such baseline model.
