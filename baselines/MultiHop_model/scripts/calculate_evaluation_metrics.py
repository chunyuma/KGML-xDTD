import sys
import os
import argparse
import pickle
import torch
import numpy as np
import utils
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## folder parameters
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default="../log_folder")
    parser.add_argument("--log_name", type=str, help="log file name", default="step15_3.log")
    parser.add_argument('--data_dir', type=str, help='The full path of data folder', default='../data')
    parser.add_argument('--drugmeshdb_match', type=str, help='The full path of file containing drugmeshdb matched paths')
    parser.add_argument('--all_paths_prob', type=str, help='The full path of all KG-based paths with probabilities')
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir,args.log_name))
    args.logger = logger
    logger.info(args)

    ## read drugmeshdb relevant data
    with open(args.drugmeshdb_match,'rb') as infile:
        match_paths = pickle.load(infile)
    with open(args.all_paths_prob,'rb') as infile:
        all_paths_prob = pickle.load(infile)

    ## read training drug-disease pairs
    train_pairs = pd.read_csv(os.path.join(args.data_dir,'pretrain_reward_shaping_model_train_val_test_random_data_3class','train_pairs.txt'), sep='\t', header=0)
    train_pairs_df = {(source, target):1 for (source, target) in train_pairs[['source','target']].to_numpy()}
    # ## read validation drug-disease pairs
    val_pairs = pd.read_csv(os.path.join(args.data_dir,'pretrain_reward_shaping_model_train_val_test_random_data_3class','val_pairs.txt'), sep='\t', header=0)
    val_pairs_df = {(source, target):1 for (source, target) in val_pairs[['source','target']].to_numpy()}

    ## read mapping files
    entity2id, id2entity = utils.load_index(os.path.join(args.data_dir, 'entity2freq.txt'))
    relation2id, id2relation = utils.load_index(os.path.join(args.data_dir, 'relation2freq.txt'))

    ## sorted paths based on probability and store them in dictionary
    all_path_dict = dict()
    filter_edges = [relation2id[edge] for edge in ['biolink:related_to','biolink:biolink:part_of','biolink:coexists_with','biolink:contraindicated_for'] if relation2id.get(edge)]
    for (source, target) in all_paths_prob:
        all_path_dict[(source, target)] = dict()
        try:
            edge_mat, node_mat, score_mat = all_paths_prob[(source, target)]
            temp = pd.DataFrame(edge_mat.numpy())
            keep_index = list(temp.loc[~(temp[1].isin(filter_edges) | temp[2].isin(filter_edges) | temp[3].isin(filter_edges)),:].index)
            edge_mat = edge_mat[keep_index]
            node_mat = node_mat[keep_index]
            score_mat = score_mat[keep_index]
            all_paths_prob[(source, target)] = [edge_mat, node_mat, score_mat]
        except:
            continue
        _, index_mat = torch.sort(score_mat, descending=True)
        edge_mat, node_mat, score_mat = edge_mat[index_mat], node_mat[index_mat], score_mat[index_mat]
        node_mat = node_mat.numpy()
        for index, x in enumerate(node_mat):
            if tuple(x) in all_path_dict[(source, target)]:
                all_path_dict[(source, target)][tuple(x)].append([tuple(edge_mat[index].numpy()), score_mat[index].item(), index+1])
            else:
                all_path_dict[(source, target)][tuple(x)] = []
                all_path_dict[(source, target)][tuple(x)].append([tuple(edge_mat[index].numpy()), score_mat[index].item(), index+1])


    ## find the matched path indexes and store them in dictionary
    match_paths_dict = dict()
    for (source, target) in match_paths:
        temp = match_paths[(source, target)][1].unique(dim=0).numpy()
        match_paths_dict[(source, target)] = dict()
        for x in temp:
            test = all_path_dict[(source, target)][tuple(x)]
            match_paths_dict[(source, target)][tuple(x)] = (sorted(test, key=lambda row: row[2])[0], len(all_paths_prob[(source, target)][2]))

    ## calculate MPR 
    percentile = []
    for x in match_paths_dict:
        # if x not in train_pairs_df:
        if x not in train_pairs_df and x not in val_pairs_df:
            temp = sorted(match_paths_dict[x].items(), key=lambda row: row[1][0][2])
            percentile += [1 - temp[0][1][0][2]/temp[0][1][1]]
    # args.logger.info(f'MPR based on non-training dataset: {np.array(percentile).mean()}')
    args.logger.info(f'MPR based on test dataset: {np.array(percentile).mean()}')


    ## calculate MRR 
    MRR = []
    for x in match_paths_dict:
        # if x not in train_pairs_df:
        if x not in train_pairs_df and x not in val_pairs_df:
            temp = sorted(match_paths_dict[x].items(), key=lambda row: row[1][0][2])
            MRR += [1/temp[0][1][0][2]]
    # args.logger.info(f'MRR based on non-training dataset: {np.array(MRR).mean()}')
    args.logger.info(f'MRR based on test dataset: {np.array(MRR).mean()}')



    ## calculate Hit@K
    for k in [1, 10, 50, 100, 500, 1000]:
        hit_at_k = []
        count = []
        for index, x in enumerate(match_paths_dict):
            # if x not in train_pairs_df:
            if x not in train_pairs_df and x not in val_pairs_df:
                temp = sorted(match_paths_dict[x].items(), key=lambda row: row[1][0][2])
                if temp[0][1][0][2] <= k:
                    hit_at_k += [index]
                count += [index]
        # args.logger.info(f'Hit@{k} based on non-training dataset: {len(hit_at_k)/len(count)}')
        args.logger.info(f'Hit@{k} based on test dataset: {len(hit_at_k)/len(count)}')
