import numpy as np
import pandas as pd
import argparse
import os
import sys
import pickle
import utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default="../log_folder")
    parser.add_argument("--log_name", type=str, help="log file name", default="step7.log")
    parser.add_argument("--data_dir", type=str, help="The path of data folder", default="../data")
    parser.add_argument("--input", type=str, help="The full path of graphsage output folder")
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir,args.log_name))
    logger.info(args)

    unsupervised_graphsage_vectors = np.load(os.path.join(args.input, 'val.npy'))
    unsupervised_graphsage_ids = pd.read_csv(open(os.path.join(args.input, 'val.txt'),'r'), header=None).rename(columns={0:'id'})
    id_vec = pd.concat([unsupervised_graphsage_ids,pd.DataFrame(unsupervised_graphsage_vectors)],axis=1)
    id_vec = id_vec.sort_values(by='id').reset_index(drop=True)
    id_vec_array = id_vec.iloc[:,1:].to_numpy()
    curie_to_ids = pd.read_csv(os.path.join(args.data_dir,'graphsage_input','id_map.txt'), sep='\t', header=0)
    unsupervised_graphsage_emb_dict = {curie:id_vec_array[index] for index, curie in enumerate(curie_to_ids['curie'])}

    ## create output folder
    if not os.path.isdir(os.path.join(args.data_dir, 'graphsage_output')):
        os.mkdir(os.path.join(args.data_dir, 'graphsage_output'))

    with open(os.path.join(args.data_dir,'graphsage_output','unsuprvised_graphsage_entity_embeddings.pkl'), 'wb') as outfile:
        pickle.dump(unsupervised_graphsage_emb_dict,outfile)




