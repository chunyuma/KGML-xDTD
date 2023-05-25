import sys, os, re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import product
from collections import Counter
import random
import math
import argparse
import utils
import pickle
import copy
from tqdm import tqdm, trange

def generate_rand_data(n, pairs, disease_list, drug_list, all_known_tp_pairs, existing_pairs=None):

    if n is not None:
        n_drug = n
        n_disease = n

    ## only use the tp data
    pairs = pairs.loc[pairs['y'] == 1,:].reset_index(drop=True)
    drug_in_data_pairs = list(set(pairs['source']))
    disease_name_list = copy.deepcopy(disease_list)
    disease_in_data_pairs = list(set(pairs['target']))
    drug_name_list = copy.deepcopy(drug_list)

    ## create a check list for all tp an tn pairs
    check_list_temp = {(all_known_tp_pairs.loc[index,'source'],all_known_tp_pairs.loc[index,'target']):1 for index in range(all_known_tp_pairs.shape[0])}
    if existing_pairs is not None:
        for index in range(existing_pairs.shape[0]):
            if (existing_pairs.loc[index,'source'],existing_pairs.loc[index,'target']) not in check_list_temp:
                check_list_temp[(existing_pairs.loc[index,'source'],existing_pairs.loc[index,'target'])] = 1

    random_pairs = []
    for drug in drug_in_data_pairs:
        if n is None:
            n_drug = len(pairs.loc[pairs['source']==drug,'y'])
        count = 0
        temp_dict = dict()
        random.shuffle(disease_name_list)
        for disease in disease_name_list:
            if (drug, disease) not in check_list_temp and (drug, disease) not in temp_dict:
                temp_dict[(drug, disease)] = 1
                count += 1
            if count == n_drug:
                break
        random_pairs += [pd.DataFrame(temp_dict.keys())]

    for disease in disease_in_data_pairs:
        if n is None:
            n_disease = len(pairs.loc[pairs['target']==disease,'y'])
        count = 0
        temp_dict = dict()
        random.shuffle(drug_name_list)
        for drug in drug_name_list:
            if (drug, disease) not in check_list_temp and (drug, disease) not in temp_dict:
                temp_dict[(drug, disease)] = 1
                count += 1
            if count == n_disease:
                break
        random_pairs += [pd.DataFrame(temp_dict.keys())]

    random_pairs = pd.concat(random_pairs).reset_index(drop=True).rename(columns={0:'source',1:'target'})
    random_pairs['y'] = 2
    
    print(f'Number of random pairs: {random_pairs.shape[0]}', flush=True)

    return random_pairs


def split_df_into_train_val_test(df, train_val_test_size, expert_demonstration_tp_pairs, data_type='tp', seed=1024):

    random_state = np.random.RandomState(seed) 

    if data_type == 'tp':
        tp_pairs = df[['source','target']].drop_duplicates()

        tp_pairs_in_expert = expert_demonstration_tp_pairs
        expert_demonstration_tp_pairs_dict = {(expert_demonstration_tp_pairs.loc[index,'source'],expert_demonstration_tp_pairs.loc[index,'target']):1 for index in range(len(expert_demonstration_tp_pairs))}
        tp_pairs_not_in_expert =  pd.DataFrame([(tp_pairs.loc[index,'source'],tp_pairs.loc[index,'target']) for index in range(len(tp_pairs)) if (tp_pairs.loc[index,'source'],tp_pairs.loc[index,'target']) not in expert_demonstration_tp_pairs_dict]).rename(columns={0:'source',1:'target'})

        ## split tp_pairs_in_expert into train set, val set and test set
        count = tp_pairs_in_expert['source'].value_counts()
        unique_temp = set(count.reset_index().loc[count.reset_index()['count']==1,'source'])
        train_tp_pairs_in_expert_1 = tp_pairs_in_expert.loc[tp_pairs_in_expert['source'].isin(unique_temp),['source','target']].reset_index(drop=True)
        rest_train_tp_pairs_in_expert = tp_pairs_in_expert.loc[~tp_pairs_in_expert['source'].isin(unique_temp),['source','target']].reset_index(drop=True)
        pad_tp_size = math.ceil(len(tp_pairs_in_expert) * train_val_test_size[0]) - len(train_tp_pairs_in_expert_1)
        val_test_size = (len(tp_pairs_in_expert) - math.ceil(len(tp_pairs_in_expert) * train_val_test_size[0]))
        curie_to_id = {source:index for index, source in enumerate(set(rest_train_tp_pairs_in_expert['source']))}
        rest_train_tp_pairs_in_expert = rest_train_tp_pairs_in_expert.apply(lambda row: [row[0],row[1],curie_to_id[row[0]]], axis=1, result_type='expand').rename(columns={0:'source', 1:'target', 2:'cluster'})

        train_pad_index, val_test_index = train_test_split(np.array(list(rest_train_tp_pairs_in_expert.index)), train_size=pad_tp_size/(pad_tp_size+val_test_size), random_state=random_state, shuffle=True, stratify=rest_train_tp_pairs_in_expert['cluster'])
        train_tp_pairs_in_expert_2 = rest_train_tp_pairs_in_expert.loc[list(train_pad_index),['source','target']].reset_index(drop=True)
        train_tp_pairs_in_expert = pd.concat([train_tp_pairs_in_expert_1,train_tp_pairs_in_expert_2]).reset_index(drop=True)
        val_test_tp_pairs_in_expert = rest_train_tp_pairs_in_expert.loc[list(val_test_index),['source','target']].reset_index(drop=True)

        val_index, test_index = train_test_split(np.array(list(val_test_tp_pairs_in_expert.index)), train_size=train_val_test_size[1]/(train_val_test_size[1]+train_val_test_size[2]), random_state=random_state, shuffle=True)
        val_tp_pairs_in_expert = val_test_tp_pairs_in_expert.loc[list(val_index),:].reset_index(drop=True)
        test_tp_pairs_in_expert = val_test_tp_pairs_in_expert.loc[list(test_index),:].reset_index(drop=True)

        train_tp_pairs_in_expert['y'] = 1
        val_tp_pairs_in_expert['y'] = 1
        test_tp_pairs_in_expert['y'] = 1

        ## split tp_pairs_not_in_expert into train set, val set and test set
        count = tp_pairs_not_in_expert['source'].value_counts()
        unique_temp = set(count.reset_index().loc[count.reset_index()['count']==1,'source'])
        train_tp_pairs_not_in_expert_1 = tp_pairs_not_in_expert.loc[tp_pairs_not_in_expert['source'].isin(unique_temp),['source','target']].reset_index(drop=True)
        rest_train_tp_pairs_not_in_expert = tp_pairs_not_in_expert.loc[~tp_pairs_not_in_expert['source'].isin(unique_temp),['source','target']].reset_index(drop=True)
        pad_tp_size = math.ceil(len(tp_pairs_not_in_expert) * train_val_test_size[0]) - len(train_tp_pairs_not_in_expert_1)
        val_test_size = (len(tp_pairs_not_in_expert) - math.ceil(len(tp_pairs_not_in_expert) * train_val_test_size[0]))
        curie_to_id = {source:index for index, source in enumerate(set(rest_train_tp_pairs_not_in_expert['source']))}
        rest_train_tp_pairs_not_in_expert = rest_train_tp_pairs_not_in_expert.apply(lambda row: [row[0],row[1],curie_to_id[row[0]]], axis=1, result_type='expand').rename(columns={0:'source', 1:'target', 2:'cluster'})

        train_pad_index, val_test_index = train_test_split(np.array(list(rest_train_tp_pairs_not_in_expert.index)), train_size=pad_tp_size/(pad_tp_size+val_test_size), random_state=random_state, shuffle=True, stratify=rest_train_tp_pairs_not_in_expert['cluster'])  
        train_tp_pairs_not_in_expert_2 = rest_train_tp_pairs_not_in_expert.loc[list(train_pad_index),['source','target']].reset_index(drop=True)
        train_tp_pairs_not_in_expert = pd.concat([train_tp_pairs_not_in_expert_1,train_tp_pairs_not_in_expert_2]).reset_index(drop=True)
        val_test_tp_pairs_not_in_expert = rest_train_tp_pairs_not_in_expert.loc[list(val_test_index),['source','target']].reset_index(drop=True)

        val_index, test_index = train_test_split(np.array(list(val_test_tp_pairs_not_in_expert.index)), train_size=train_val_test_size[1]/(train_val_test_size[1]+train_val_test_size[2]), random_state=random_state, shuffle=True)
        val_tp_pairs_not_in_expert = val_test_tp_pairs_not_in_expert.loc[list(val_index),:].reset_index(drop=True)
        test_tp_pairs_not_in_expert = val_test_tp_pairs_not_in_expert.loc[list(test_index),:].reset_index(drop=True)

        train_tp_pairs_not_in_expert['y'] = 1
        val_tp_pairs_not_in_expert['y'] = 1
        test_tp_pairs_not_in_expert['y'] = 1

        return [(train_tp_pairs_in_expert, train_tp_pairs_not_in_expert), (val_tp_pairs_in_expert, val_tp_pairs_not_in_expert), (test_tp_pairs_in_expert, test_tp_pairs_not_in_expert)]

    else:

        tn_pairs = df[['source','target']].drop_duplicates()

        ## split triples based on the ratio of train, valid and test sets
        count = tn_pairs['source'].value_counts()
        unique_temp = set(count.reset_index().loc[count.reset_index()['count']==1,'source'])
        train_tn_pairs_1 = tn_pairs.loc[tn_pairs['source'].isin(unique_temp),['source','target']].reset_index(drop=True)
        rest_train_tn_pairs = tn_pairs.loc[~tn_pairs['source'].isin(unique_temp),['source','target']].reset_index(drop=True)
        pad_tp_size = math.ceil(len(tn_pairs) * train_val_test_size[0]) - len(train_tn_pairs_1)
        val_test_size = (len(tn_pairs) - math.ceil(len(tn_pairs) * train_val_test_size[0]))
        curie_to_id = {source:index for index, source in enumerate(set(rest_train_tn_pairs['source']))}
        rest_train_tn_pairs = rest_train_tn_pairs.apply(lambda row: [row[0],row[1],curie_to_id[row[0]]], axis=1, result_type='expand').rename(columns={0:'source', 1:'target', 2:'cluster'})

        train_pad_index, val_test_index = train_test_split(np.array(list(rest_train_tn_pairs.index)), train_size=pad_tp_size/(pad_tp_size+val_test_size), random_state=random_state, shuffle=True, stratify=rest_train_tn_pairs['cluster'])        
        train_tn_pairs_2 = rest_train_tn_pairs.loc[list(train_pad_index),['source','target']].reset_index(drop=True)
        train_tn_pairs = pd.concat([train_tn_pairs_1,train_tn_pairs_2]).reset_index(drop=True)
        val_test_tn_pairs = rest_train_tn_pairs.loc[list(val_test_index),['source','target']].reset_index(drop=True)

        val_index, test_index = train_test_split(np.array(list(val_test_tn_pairs.index)), train_size=train_val_test_size[1]/(train_val_test_size[1]+train_val_test_size[2]), random_state=random_state, shuffle=True)
        val_tn_pairs = val_test_tn_pairs.loc[list(val_index),:].reset_index(drop=True)
        test_tn_pairs = val_test_tn_pairs.loc[list(test_index),:].reset_index(drop=True)

        train_tn_pairs['y'] = 0
        val_tn_pairs['y'] = 0
        test_tn_pairs['y'] = 0

        return [train_tn_pairs, val_tn_pairs, test_tn_pairs]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default="~/DTD_RL_model/log_folder")
    parser.add_argument("--log_name", type=str, help="log file name", default="step4.log")
    parser.add_argument('--data_dir', type=str, help='Full path of data folder', default='~/data')
    parser.add_argument("--n_random", type=int, help="Number of random pairs assigned to each TP drug in train set, val set and test set for accuracy and f1score", default=30)
    parser.add_argument("--n_random_test_mrr_hk", type=int, help="Number of random pairs assigned to each TP drug in test set for MRR and H@K", default=500)
    parser.add_argument("--train_val_test_size", type=str, help="Proportion of training data, validation data and test data", default="[0.8, 0.1, 0.1]")
    parser.add_argument('--expert_dir_name', type=str, help='The name of expert path directory', default='expert_path_files')
    parser.add_argument('--path_file_name', type=str, default='expert_demonstration_relation_entity_max3.pkl', help='expert demonstration path file name')
    parser.add_argument('--seed', type=int, help='Random seed (default: 1023)', default=1023)
    parser.add_argument('--max_path', type=int, help='Maximum length of path', default=3)
    args = parser.parse_args()

    random.seed(args.seed)
    train_val_test_size = eval(args.train_val_test_size)
    class InputError(Exception):
        pass
    if not sum(train_val_test_size)==1:
        raise InputError("The sum of percents in train_val_test_size should be 1")

    logger = utils.get_logger(os.path.join(args.log_dir,args.log_name))
    logger.info(args)

    triples_without_tp_tn_triples = pd.read_csv(os.path.join(args.data_dir, 'graph_edges.txt'), sep='\t', header=0)
    triples_without_tp_tn_triples = triples_without_tp_tn_triples[['source','target','predicate']]
    triple_size = len(triples_without_tp_tn_triples)
    triples_without_tp_tn_triples = triples_without_tp_tn_triples.drop_duplicates().reset_index(drop=True)
    if len(triples_without_tp_tn_triples) < triple_size:
        triples_without_tp_tn_triples.to_csv(os.path.join(args.data_dir,'graph_edges.txt'), sep='\t', index=None)
    tp_triples = pd.read_csv(os.path.join(args.data_dir, 'tp_pairs.txt'), sep='\t', header=0)
    tp_triples['predicate'] = 'biolink:has_effect'
    tn_triples = pd.read_csv(os.path.join(args.data_dir, 'tn_pairs.txt'), sep='\t', header=0)
    tn_triples['predicate'] = 'biolink:has_no_effect'
    all_triples = pd.concat([triples_without_tp_tn_triples,tp_triples,tn_triples]).reset_index(drop=True)
    all_triples = all_triples.drop_duplicates()
    all_nodes = set()
    all_nodes.update(set(all_triples.source))
    all_nodes.update(set(all_triples.target))
    all_triples.to_csv(os.path.join(args.data_dir, 'all_triples.txt'), sep='\t', index=None)

    ## find all disease ids
    entity2id, id2entity = utils.load_index(os.path.join(args.data_dir, 'entity2freq.txt'))
    type2id, id2type = utils.load_index(os.path.join(args.data_dir, 'type2freq.txt'))
    with open(os.path.join(args.data_dir, 'entity2typeid.pkl'), 'rb') as infile:
        entity2typeid = pickle.load(infile)
    disease_type = ['biolink:Disease', 'biolink:PhenotypicFeature', 'biolink:BehavioralFeature', 'biolink:DiseaseOrPhenotypicFeature']
    disease_type_ids = [type2id[x] for x in disease_type]
    disease_names = [id2entity[index] for index, typeid in enumerate(entity2typeid) if typeid in disease_type_ids]
    drug_type = ['biolink:Drug', 'biolink:SmallMolecule']
    drug_type_ids = [type2id[x] for x in drug_type]
    drug_names = [id2entity[index] for index, typeid in enumerate(entity2typeid) if typeid in drug_type_ids]
    all_p_tp_pairs = pd.read_csv(os.path.join(args.data_dir, "all_known_tps.txt"), sep='\t', header=0)
    all_known_tp_pairs =  all_p_tp_pairs.drop_duplicates().reset_index(drop=True)

    ## save the 'treat' and 'not_treat' triples as train triples, valid triples and test triples
    ## 'treat'
    with open(os.path.join(args.data_dir, args.expert_dir_name, f"expert_demonstration_paths_max{args.max_path}_filtered.pkl"), 'rb') as infile:
        temp = pickle.load(infile)
    expert_demonstration_tp_pairs = pd.DataFrame(temp.keys(), columns=['source','target'])
    (train_tp_pairs_in_expert, train_tp_pairs_not_in_expert), (val_tp_pairs_in_expert, val_tp_pairs_not_in_expert), (test_tp_pairs_in_expert, test_tp_pairs_not_in_expert) = split_df_into_train_val_test(tp_triples, train_val_test_size, expert_demonstration_tp_pairs, data_type='tp', seed=args.seed)

    ## 'not treat'
    train_tn_pairs, val_tn_pairs, test_tn_pairs = split_df_into_train_val_test(tn_triples, train_val_test_size, None, 'tn', seed=args.seed)

    pretrain_data_train_pairs = pd.concat([train_tp_pairs_in_expert, train_tp_pairs_not_in_expert, train_tn_pairs]).reset_index(drop=True)
    pretrain_data_train_pairs = pretrain_data_train_pairs.sample(frac=1).reset_index(drop=True)
    pretrain_data_val_pairs = pd.concat([val_tp_pairs_in_expert, val_tp_pairs_not_in_expert, val_tn_pairs]).reset_index(drop=True)
    pretrain_data_val_pairs = pretrain_data_val_pairs.sample(frac=1).reset_index(drop=True)
    pretrain_data_test_pairs = pd.concat([test_tp_pairs_in_expert, test_tp_pairs_not_in_expert, test_tn_pairs]).reset_index(drop=True)
    pretrain_data_test_pairs = pretrain_data_test_pairs.sample(frac=1).reset_index(drop=True)

    existing_pairs = pd.concat([pretrain_data_train_pairs,pretrain_data_val_pairs,pretrain_data_test_pairs]).reset_index(drop=True)
    random_pairs = generate_rand_data(None, pretrain_data_train_pairs, disease_names, drug_names, all_known_tp_pairs, existing_pairs)
    pretrain_data_train_pairs = pd.concat([pretrain_data_train_pairs,random_pairs]).reset_index(drop=True).sample(frac=1).reset_index(drop=True)

    existing_pairs = pd.concat([pretrain_data_train_pairs,pretrain_data_val_pairs,pretrain_data_test_pairs]).reset_index(drop=True)
    random_pairs = generate_rand_data(None, pretrain_data_val_pairs, disease_names, drug_names, all_known_tp_pairs, existing_pairs)
    pretrain_data_val_pairs = pd.concat([pretrain_data_val_pairs,random_pairs]).reset_index(drop=True).sample(frac=1).reset_index(drop=True)

    existing_pairs = pd.concat([pretrain_data_train_pairs,pretrain_data_val_pairs,pretrain_data_test_pairs]).reset_index(drop=True)
    random_pairs = generate_rand_data(None, pretrain_data_test_pairs, disease_names, drug_names, all_known_tp_pairs, existing_pairs)
    pretrain_data_test_pairs = pd.concat([pretrain_data_test_pairs,random_pairs]).reset_index(drop=True).sample(frac=1).reset_index(drop=True)

    existing_pairs = pd.concat([pretrain_data_train_pairs,pretrain_data_val_pairs,pretrain_data_test_pairs]).reset_index(drop=True)
    pretrain_data_val_test_pairs = pd.concat([pretrain_data_val_pairs,pretrain_data_test_pairs]).reset_index(drop=True)
    val_test_random_pairs = generate_rand_data(args.n_random_test_mrr_hk, pretrain_data_val_test_pairs, disease_names, drug_names, all_known_tp_pairs, existing_pairs)

    args.pretrain_outdir_3class = os.path.join(args.data_dir,'pretrain_reward_shaping_model_train_val_test_random_data_3class')
    if not os.path.isdir(args.pretrain_outdir_3class):
        os.mkdir(args.pretrain_outdir_3class)

    pretrain_data_train_pairs.to_csv(os.path.join(args.pretrain_outdir_3class, 'train_pairs.txt'), sep='\t', index=None)
    pretrain_data_val_pairs.to_csv(os.path.join(args.pretrain_outdir_3class, 'val_pairs.txt'), sep='\t', index=None)
    pretrain_data_test_pairs.to_csv(os.path.join(args.pretrain_outdir_3class, 'test_pairs.txt'), sep='\t', index=None)
    val_test_random_pairs.to_csv(os.path.join(args.pretrain_outdir_3class, 'random_pairs.txt'), sep='\t', index=None)

    args.pretrain_outdir_2class = os.path.join(args.data_dir,'pretrain_reward_shaping_model_train_val_test_random_data_2class')
    if not os.path.isdir(args.pretrain_outdir_2class):
        os.makedirs(args.pretrain_outdir_2class)

    args.ten_times_random_pairs = os.path.join(args.pretrain_outdir_2class, '10times_random_pairs')
    if not os.path.isdir(args.ten_times_random_pairs):
        os.makedirs(args.ten_times_random_pairs)

    ## generate 10 times random pairs for test set
    for i in trange(10):
        existing_pairs = pd.concat([pretrain_data_train_pairs,pretrain_data_val_pairs,pretrain_data_test_pairs]).reset_index(drop=True)
        random_pairs = generate_rand_data(args.n_random_test_mrr_hk, pretrain_data_test_pairs, disease_names, drug_names, all_known_tp_pairs, existing_pairs)
        random_pairs.to_csv(os.path.join(args.ten_times_random_pairs, 'random_pairs_{}.txt'.format(i)), sep='\t', index=None)

    pretrain_data_train_pairs.loc[pretrain_data_train_pairs['y']!=2,:].reset_index(drop=True).to_csv(os.path.join(args.pretrain_outdir_2class, 'train_pairs.txt'), sep='\t', index=None)
    pretrain_data_val_pairs.loc[pretrain_data_val_pairs['y']!=2,:].reset_index(drop=True).to_csv(os.path.join(args.pretrain_outdir_2class, 'val_pairs.txt'), sep='\t', index=None)
    pretrain_data_test_pairs.loc[pretrain_data_test_pairs['y']!=2,:].reset_index(drop=True).to_csv(os.path.join(args.pretrain_outdir_2class, 'test_pairs.txt'), sep='\t', index=None)
    val_test_random_pairs.to_csv(os.path.join(args.pretrain_outdir_2class, 'random_pairs.txt'), sep='\t', index=None)

    ## generate train set, val set and test set for RL model
    args.rl_outdir = os.path.join(args.data_dir,'RL_model_train_val_test_data')
    if not os.path.isdir(args.rl_outdir):
        os.makedirs(args.rl_outdir)

    rl_train_tp_pairs = train_tp_pairs_in_expert
    rl_val_tp_pairs = val_tp_pairs_in_expert
    rl_test_tp_pairs = test_tp_pairs_in_expert
    rl_tp_pairs = pd.concat([rl_train_tp_pairs,rl_val_tp_pairs,rl_test_tp_pairs]).reset_index(drop=True)

    rl_train_tp_pairs.to_csv(os.path.join(args.rl_outdir, 'train_pairs.txt'), sep='\t', index=None)
    rl_val_tp_pairs.to_csv(os.path.join(args.rl_outdir, 'val_pairs.txt'), sep='\t', index=None)
    rl_test_tp_pairs.to_csv(os.path.join(args.rl_outdir, 'test_pairs.txt'), sep='\t', index=None)
    rl_tp_pairs.to_csv(os.path.join(args.rl_outdir, 'all_pairs.txt'), sep='\t', index=None)

    ## split expert paths based on train set, validation set and test set
    with open(os.path.join(args.data_dir, args.expert_dir_name, f'expert_demonstration_relation_entity_max{args.max_path}_filtered.pkl'),'rb') as infile:
        expert_demonstration_relation_entity = pickle.load(infile)

    expert_demonstration_relation_entity_df = pd.DataFrame(expert_demonstration_relation_entity[1].numpy())

    train_temp = rl_train_tp_pairs.apply(lambda row: [entity2id[row[0]],entity2id[row[1]]], axis=1, result_type='expand')
    id_list = [list(expert_demonstration_relation_entity_df.loc[(expert_demonstration_relation_entity_df[0]==train_temp.loc[index,0]) & (expert_demonstration_relation_entity_df[3]==train_temp.loc[index,1]),:].index) for index in range(len(train_temp))]
    ids = [y for x in id_list for y in x]
    train_expert_demonstration_relation_entity = [expert_demonstration_relation_entity[0][ids], expert_demonstration_relation_entity[1][ids]]
    with open(os.path.join(args.data_dir, args.expert_dir_name, f'train_expert_demonstration_relation_entity_max{args.max_path}_filtered.pkl'),'wb') as outfile:
        pickle.dump(train_expert_demonstration_relation_entity, outfile)

    val_temp = rl_val_tp_pairs.apply(lambda row: [entity2id[row[0]],entity2id[row[1]]], axis=1, result_type='expand')
    id_list = [list(expert_demonstration_relation_entity_df.loc[(expert_demonstration_relation_entity_df[0]==val_temp.loc[index,0]) & (expert_demonstration_relation_entity_df[3]==val_temp.loc[index,1]),:].index) for index in range(len(val_temp))]
    ids = [y for x in id_list for y in x]
    val_expert_demonstration_relation_entity = [expert_demonstration_relation_entity[0][ids], expert_demonstration_relation_entity[1][ids]]
    with open(os.path.join(args.data_dir, args.expert_dir_name, f'val_expert_demonstration_relation_entity_max{args.max_path}_filtered.pkl'),'wb') as outfile:
        pickle.dump(val_expert_demonstration_relation_entity, outfile)

    test_temp = rl_test_tp_pairs.apply(lambda row: [entity2id[row[0]],entity2id[row[1]]], axis=1, result_type='expand')
    id_list = [list(expert_demonstration_relation_entity_df.loc[(expert_demonstration_relation_entity_df[0]==test_temp.loc[index,0]) & (expert_demonstration_relation_entity_df[3]==test_temp.loc[index,1]),:].index) for index in range(len(test_temp))]
    ids = [y for x in id_list for y in x]
    test_expert_demonstration_relation_entity = [expert_demonstration_relation_entity[0][ids], expert_demonstration_relation_entity[1][ids]]
    with open(os.path.join(args.data_dir, args.expert_dir_name, f'test_expert_demonstration_relation_entity_max{args.max_path}_filtered.pkl'),'wb') as outfile:
        pickle.dump(test_expert_demonstration_relation_entity, outfile)
