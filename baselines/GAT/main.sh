
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

## train GAT model
echo 'train GAT baseline model'
python ${work_folder}/scripts/run_model.py --log_dir ${work_folder}/log_folder \
                                               --log_name additional_GAT_run.log \
                                               --data_dir ${work_folder}/data \
                                               --processed_data_dir ${work_folder}/data/GAT_processed_data \
                                               --use_gpu \
                                               --gpu 0 \
                                               --learning_ratio 0.001 \
                                               --init_emb_size 100 \
                                               --use_known_embedding \
                                               --known_embedding ${work_folder}/data/embedding_biobert_namecat.pkl \
                                               --num_epochs 500 \
                                               --emb_size 128 \
                                               --num_head 3 \
                                               --batch_size 500 \
                                               --num_layers 2 \
                                               --output_folder ${work_folder}/models
