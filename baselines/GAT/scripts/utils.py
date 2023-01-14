import pickle
from datetime import timedelta
from torch.utils.data import Dataset
from torch import tensor
import matplotlib.pyplot as plt
import pandas as pd
import re
import sys
import logging
import random
import torch
import numpy as np
from sklearn.metrics import f1_score
from torch_geometric.data import NeighborSampler
plt.switch_backend('agg')

def calculate_f1score(preds, labels, threshold=0.5, mode='binary'):
    preds = np.array(preds)
    labels = np.array(labels)
    preds = (np.array(preds)>threshold).astype(int)
    f1score = f1_score(labels, preds, average=mode)
    return f1score

def calculate_acc(preds, labels, threshold=0.5):
    preds = np.array(preds)
    labels = np.array(labels)
    preds = (np.array(preds)>threshold).astype(int)
    acc = (preds == labels).astype(float).mean()
    return acc


def calculate_mrr(drug_disease_pairs, random_pairs, N):
    '''
    This function is used to calculate Mean Reciprocal Rank (MRR)
    reference paper: Knowledge Graph Embedding for Link Prediction: A Comparative Analysis
    '''

    ## only use tp pairs
    drug_disease_pairs = drug_disease_pairs.loc[drug_disease_pairs['y']==1,:].reset_index(drop=True)

    Q_n = len(drug_disease_pairs)
    score = 0
    for index in range(Q_n):
        query_drug = drug_disease_pairs['source'][index]
        query_disease = drug_disease_pairs['target'][index]
        this_query_score = drug_disease_pairs['prob'][index]
        all_random_probs_for_this_query = list(random_pairs.loc[random_pairs['source'].isin([query_drug]),'prob'])
        all_random_probs_for_this_query += list(random_pairs.loc[random_pairs['target'].isin([query_disease]),'prob'])
        all_random_probs_for_this_query = all_random_probs_for_this_query[:N]
        all_in_list = [this_query_score] + all_random_probs_for_this_query
        rank = list(torch.tensor(all_in_list).sort(descending=True).indices.numpy()).index(0)+1
        score += 1/rank

    final_score = score/Q_n

    return final_score

def calculate_hitk(drug_disease_pairs, random_pairs, N, k=1):
    '''
    This function is used to calculate Hits@K (H@K)
    reference paper: Knowledge Graph Embedding for Link Prediction: A Comparative Analysis
    '''

    ## only use tp pairs
    drug_disease_pairs = drug_disease_pairs.loc[drug_disease_pairs['y']==1,:].reset_index(drop=True)

    Q_n = len(drug_disease_pairs)
    count = 0
    for index in range(Q_n):
        query_drug = drug_disease_pairs['source'][index]
        query_disease = drug_disease_pairs['target'][index]
        this_query_score = drug_disease_pairs['prob'][index]
        all_random_probs_for_this_query = list(random_pairs.loc[random_pairs['source'].isin([query_drug]),'prob'])
        all_random_probs_for_this_query += list(random_pairs.loc[random_pairs['target'].isin([query_disease]),'prob'])
        all_random_probs_for_this_query = all_random_probs_for_this_query[:N]
        all_in_list = [this_query_score] + all_random_probs_for_this_query
        rank = list(torch.tensor(all_in_list).sort(descending=True).indices.numpy()).index(0)+1
        if rank <= k:
            count += 1

    final_score = count/Q_n

    return final_score


def calculate_rank(data_pos_df, all_drug_ids, all_disease_ids, all_tp_pairs_dict, data, model, args, mode='both'):
    res_dict = dict()
    total = data_pos_df.shape[0]

    for _, n_id, adjs in NeighborSampler(data.edge_index, sizes=args.layer_size, batch_size=data.feat.shape[0], shuffle=False, num_workers=8):
        adjs = [(adj.edge_index,adj.size) for adj in adjs]
        n_id = n_id.to(args.device)
        # adjs = [(adj[0],(int(adj[1][0]),int(adj[1][1]))) for adj in adjs]
        x_n_id = data.feat[n_id]
        all_embeddings = model.get_gnn_embedding(x_n_id, adjs, n_id)
        break


    for index, (source, target) in enumerate(data_pos_df[['source','target']].to_numpy()):
        print(f"calculating rank {index+1}/{total}", flush=True)
        this_pair = source + '_' + target
        temp_df = pd.concat([pd.DataFrame(zip(all_drug_ids,[target]*len(all_drug_ids))),pd.DataFrame(zip([source]*len(all_disease_ids),all_disease_ids))]).reset_index(drop=True)
        link =temp_df[[0,1]].apply(lambda row: [args.entity2id.get(row[0]) - 1, args.entity2id.get(row[1]) - 1], axis=1, result_type='expand').rename(columns={0: "source", 1: "target"})
        link = torch.tensor(np.array(link), dtype=torch.long).to(args.device)
        temp_df[2] = temp_df[0] + '_' + temp_df[1]
        temp_df[3] = model.predict2(all_embeddings, link, n_id)
        this_row = temp_df.loc[temp_df[2]==this_pair,:].reset_index(drop=True).iloc[[0]]
        temp_df = temp_df.loc[temp_df[2]!=this_pair,:].reset_index(drop=True)
        if mode == 'both':
            ## without filter
            # (1) for drug
            temp_df_1 = pd.concat([temp_df.loc[temp_df[1] == target,:],this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            w_drug_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            # (2) for disease
            temp_df_1 = pd.concat([temp_df.loc[temp_df[0] == source,:],this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            w_disease_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            # (3) for both 
            temp_df_1 = pd.concat([temp_df,this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            w_both_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            ## filter 
            temp_df = temp_df.loc[~temp_df[2].isin(list(all_tp_pairs_dict.keys())),:].reset_index(drop=True)
             # (1) for drug
            temp_df_1 = pd.concat([temp_df.loc[temp_df[1] == target,:],this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            drug_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            # (2) for disease
            temp_df_1 = pd.concat([temp_df.loc[temp_df[0] == source,:],this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            disease_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            # (3) for both 
            temp_df_1 = pd.concat([temp_df,this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            both_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            res_dict[(source, target)] = [(w_drug_rank, w_disease_rank, w_both_rank), (drug_rank, disease_rank, both_rank)]

        elif mode == 'filter':
            ## filter 
            temp_df = temp_df.loc[~temp_df[2].isin(list(all_tp_pairs_dict.keys())),:].reset_index(drop=True)
             # (1) for drug
            temp_df_1 = pd.concat([temp_df.loc[temp_df[1] == target,:],this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            drug_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            # (2) for disease
            temp_df_1 = pd.concat([temp_df.loc[temp_df[0] == source,:],this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            disease_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            # (3) for both 
            temp_df_1 = pd.concat([temp_df,this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            both_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            res_dict[(source, target)] = [(None, None, None), (drug_rank, disease_rank, both_rank)] 
        else:
            ## without filter
            # (1) for drug
            temp_df_1 = pd.concat([temp_df.loc[temp_df[1] == target,:],this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            w_drug_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            # (2) for disease
            temp_df_1 = pd.concat([temp_df.loc[temp_df[0] == source,:],this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            w_disease_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            # (3) for both 
            temp_df_1 = pd.concat([temp_df,this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            w_both_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            res_dict[(source, target)] = [(w_drug_rank, w_disease_rank, w_both_rank), (None, None, None)]
    return res_dict

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(timedelta(seconds=elapsed_rounded))

def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s  [%(levelname)s]  %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def load_index(input_path):
    name_to_id, id_to_name = {}, {}
    with open(input_path) as f:
        for index, line in enumerate(f.readlines()):
            name, _ = line.strip().split()
            name_to_id[name] = index
            id_to_name[index] = name
    return name_to_id, id_to_name

class DataWrapper(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        n_id, adjs = pickle.load(open(self.paths[idx],'rb'))
        return (n_id, adjs)

def empty_gpu_cache(args):
    with torch.cuda.device(f'cuda:{args.gpu}'):
        torch.cuda.empty_cache()