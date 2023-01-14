#!/bin/bash

## set up current path
work_folder=$(pwd)

## create required folders
if [ ! -d "${work_folder}/data" ]
then
		ln -s ../../data
fi
if [ ! -d "${work_folder}/results" ]
then
		mkdir ${work_folder}/results
fi

## 1. Preprocess data
echo "preprocess data"
GRAPH_DATA_PATH=${work_folder}/data/
PAIR_DATA_PATH=${work_folder}/data/pretrain_reward_shaping_model_train_val_test_random_data_2class/
OUTPUT_PATH=${work_folder}/results/processed/

python ${work_folder}/scripts/preprocess.py --graph_data_path ${GRAPH_DATA_PATH} --pair_data_path ${PAIR_DATA_PATH} --output_path ${OUTPUT_PATH}


## 2. Train the models
echo "train the models"
CHECKPOINT_PATH=${work_folder}/results/checkpoints/
python ${work_folder}/scripts/train.py --model transe --dim 100 --batch_size 10000 --epoch 1000 --data ${OUTPUT_PATH} --checkpoint ${CHECKPOINT_PATH}
python ${work_folder}/scripts/train.py --model transr --dim 50 --batch_size 2000 --epoch 1000 --data ${OUTPUT_PATH} --checkpoint ${CHECKPOINT_PATH}
python ${work_folder}/scripts/train.py --model rotate --dim 30 --batch_size 2000 --epoch 1000 --data ${OUTPUT_PATH} --checkpoint ${CHECKPOINT_PATH}
python ${work_folder}/scripts/train.py --model distmult --dim 100 --batch_size 10000 --epoch 1000 --data ${OUTPUT_PATH} --checkpoint ${CHECKPOINT_PATH}
python ${work_folder}/scripts/train.py --model complex --dim 50 --batch_size 2000 --epoch 500 --data ${OUTPUT_PATH} --checkpoint ${CHECKPOINT_PATH}
python ${work_folder}/scripts/train.py --model analogy --dim 20 --batch_size 2000 --epoch 500 --data ${OUTPUT_PATH} --checkpoint ${CHECKPOINT_PATH}
python ${work_folder}/scripts/train.py --model simple --dim 100 --batch_size 2000 --epoch 500 --data ${OUTPUT_PATH} --checkpoint ${CHECKPOINT_PATH}

## 3. Evaluate the models
echo "evaluate the models"
python ${work_folder}/scripts/evaluate.py --data ${OUTPUT_PATH} --pair_data_path ${PAIR_DATA_PATH} --checkpoint ${CHECKPOINT_PATH}