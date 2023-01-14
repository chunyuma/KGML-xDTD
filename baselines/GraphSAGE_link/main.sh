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


echo 'train GraphSAGE_link baseline model'
python ${work_folder}/scripts/run_model.py --data_path ${work_folder}/data \
											   --result_folder ${work_folder}/results \
											   --use_known_embedding \
											   --init_emb_file ${work_folder}/data/embedding_biobert_namecat.pkl \
											   --num_layers 2 \
											   --layer_size 96 96 \
											   --dropout_p 0 \
											   --use_gpu \
											   --gpu 0 \
											   --learning_ratio 0.001 \
											   --num_epochs 1000 \
											   --batch_size 512 \
											   --print_every 1

