#!/usr/bin/env bash

## set working directory
work_folder=$(pwd)

## create required folders
if [ ! -d "${work_folder}/data" ]
then
		ln -s ../../data
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
if [ ! -d "${work_folder}/drugmeshdb" ]
then
		ln -s ../../drugmeshdb
		mkdir ${work_folder}/drugmeshdb
fi

# link data
cd ${work_folder}
cd ${work_folder}/scripts
ln -s ../../../bkg_rtxkg2c_v2.7.3/relevant_dbs/node_synonymizer_v1.0_KG2.7.3.sqlite

# step1: train RL model
echo 'running step1: train RL model'
python ${work_folder}/scripts/run_RL_model.py --log_dir ${work_folder}/log_folder --log_name step1.log --data_dir ${work_folder}/data --output_folder ${work_folder} --pretrain_model_path ${work_folder}/models/RF_model_3class/RF_model.pt --use_init_emb --use_gpu --lr 0.0005 --num_rollout_steps 3 --history_dim 256 --reward_shaping_threshold 0.35 --action_dropout_rate 0.5 --num_peek_epochs 2 --num_wait_epochs 1000 --num_check_epochs 20 --train_batch_size 32 --eval_batch_size 64 --num_epochs 1000 --beta 0.005 --beam_size 50 --num_rollouts 35 --gamma 0.99 --factor 0.9

# step2: evaluate the predicted paths
echo 'running step2: evaluate the predicted paths'
python ${work_folder}/scripts/calculate_path_prob.py --log_dir ${work_folder}/log_folder --log_name step2_1.log --data_dir ${work_folder}/data --target_paths_file ${work_folder}/drugmeshdb/res_all_paths.pkl --output_file ${work_folder}/drugmeshdb/res_all_paths_prob.pkl --max_path 3 --bandwidth 3000 --bucket_interval 50 --pretrain_model_path ${work_folder}/models/RF_model_3class/RF_model.pt --use_init_emb --use_gpu --history_dim 256 --batch_size 5000 --eval_batch_size 64 --beam_size 50 --factor 0.9
python ${work_folder}/scripts/calculate_evaluation_metrics.py --log_dir ${work_folder}/log_folder --log_name step2_2.log --data_dir ${work_folder}/data --drugmeshdb_match ${work_folder}/drugmeshdb/match_paths.pkl --all_paths_prob ${work_folder}/drugmeshdb/res_all_paths_prob.pkl
