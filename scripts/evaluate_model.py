import sys
import os
import argparse
import pickle
import torch
import torch.optim as optim
import numpy as np
import utils
import joblib
import math
import time
import pandas as pd
from tqdm import tqdm
from knowledge_graph import KnowledgeGraph
from kg_env import KGEnvironment
from models import DiscriminatorActorCritic, Transition
from random import sample, choices
from hummingbird.ml import convert
from glob import glob

def evaluate_model(args):
    kg = KnowledgeGraph(args, bandwidth=args.bandwidth, entity_dim=args.entity_dim, entity_type_dim=args.entity_type_dim, relation_dim=args.relation_dim, emb_dropout_rate=args.emb_dropout_rate, bucket_interval=args.bucket_interval, load_graph=True)
    ## read pre-train model
    pretrain_model = joblib.load(args.pretrain_model_path)
    ## convert sklearn model to pytorch
    pretrain_model = convert(pretrain_model, 'pytorch')
    if args.use_gpu is True:
        pretrain_model.to(f"cuda:{args.gpu}")
    env = KGEnvironment(args, pretrain_model, kg, max_path_len=args.max_path, state_pre_history=args.state_history)

    ## read evaluation dataset (val and test datasets) for ADAC model
    train_pairs = pd.read_csv(os.path.join(args.data_dir,'RL_model_train_val_test_data', 'train_pairs.txt'), sep='\t', header=0)
    train_pairs = train_pairs.apply(lambda row: [kg.entity2id[row[0]],kg.entity2id[row[1]]], axis=1, result_type='expand')
    train_drug_disease_dict = {did:list(set(train_pairs.loc[train_pairs[0]==did,1])) for did in set(train_pairs[0])} 
    val_pairs = pd.read_csv(os.path.join(args.data_dir,'RL_model_train_val_test_data', 'val_pairs.txt'), sep='\t', header=0)
    val_pairs = val_pairs.apply(lambda row: [kg.entity2id[row[0]],kg.entity2id[row[1]]], axis=1, result_type='expand')
    test_pairs = pd.read_csv(os.path.join(args.data_dir,'RL_model_train_val_test_data', 'test_pairs.txt'), sep='\t', header=0)
    test_pairs = test_pairs.apply(lambda row: [kg.entity2id[row[0]],kg.entity2id[row[1]]], axis=1, result_type='expand')
    eval_pairs = val_pairs
    # eval_pairs = pd.concat([val_pairs,test_pairs]).reset_index(drop=True)
    eval_drug_disease_dict = {did:list(set(eval_pairs.loc[eval_pairs[0]==did,1])) for did in set(eval_pairs[0])}
    all_pairs = pd.read_csv(os.path.join(args.data_dir,'RL_model_train_val_test_data', 'all_pairs.txt'), sep='\t', header=0)
    all_pairs = all_pairs.apply(lambda row: [kg.entity2id[row[0]],kg.entity2id[row[1]]], axis=1, result_type='expand')
    all_drug_disease_dict = {did:list(set(all_pairs.loc[all_pairs[0]==did,1])) for did in set(all_pairs[0])} 

    ## set up ADAC model
    sorted_model_files = sorted(glob(os.path.join(args.policy_net_folder,'*')),key=lambda path: int(os.path.basename(path).split('.')[0].replace('policy_model_epoch','')))
    for path in sorted_model_files:
        this_epoch = os.path.basename(path).split('.')[0].split('_')[-1]
        args.logger.info(f"Evaluating the model from {this_epoch}")
        model = DiscriminatorActorCritic(args, kg, args.state_history, args.gamma, args.target_update, args.ac_hidden, args.disc_hidden, args.metadisc_hidden)
        policy_net = torch.load(path, map_location=args.device)
        model_temp = model.policy_net.state_dict()
        model_temp.update(policy_net)
        model.policy_net.load_state_dict(model_temp)

        env.reset()
        args.logger.info(f"Evaluate model with train dataset:")
        results = utils.evaluate(args, train_drug_disease_dict, env, model, all_drug_disease_dict, save_paths=args.save_pred_paths)
        if args.save_pred_paths:
            with open(os.path.join(args.pred_paths_save_location,'train_pred_paths.pkl'),'wb') as outfile:
                pickle.dump(results, outfile)
        env.reset()
        args.logger.info(f"Evaluate model with test dataset:")
        results = utils.evaluate(args, eval_drug_disease_dict, env, model, all_drug_disease_dict, save_paths=args.save_pred_paths)
        if args.save_pred_paths:
            with open(os.path.join(args.pred_paths_save_location,'eval_pred_paths.pkl'),'wb') as outfile:
                pickle.dump(results, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## folder parameters
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default="../log_folder")
    parser.add_argument("--log_name", type=str, help="log file name", default="step12.log")
    parser.add_argument('--data_dir', type=str, help='Full path of data folder', default='../data')
    parser.add_argument('--policy_net_folder', type=str, help='The full path of folder containing trained policy network model', default='../models/ADAC_model/policy_net')
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default="../")

    ## knowledge graph and environment parameters
    parser.add_argument('--entity_dim', type=int, help='Dimension of entity embedding', default=100)
    parser.add_argument('--relation_dim', type=int, help='Dimension of relation embedding', default=100)
    parser.add_argument('--entity_type_dim', type=int, help='Dimension of entity type embedding', default=100)
    parser.add_argument('--max_path', type=int, help='Maximum length of path', default=3)
    parser.add_argument('--bandwidth', type=int, help='Maximum number of neighbors', default=3000)
    parser.add_argument('--bucket_interval', type=int, help='adjacency list bucket size to save memory (default: 50)', default=50)
    parser.add_argument('--state_history', type=int, help='state history length', default=1)
    parser.add_argument("--emb_dropout_rate", type=float, help="Knowledge entity and relation embedding vector dropout rate (default: 0)", default=0)
    parser.add_argument("--pretrain_model_path", type=str, help="The path of pretrain model", default='../models/RF_model/RF_model.pt')
    parser.add_argument('--tp_reward', type=float, help='Reward if the agent hits the target entity (default: 1.0)', default=1.0)

    # discriminator parameters
    parser.add_argument('--disc_hidden', type=int, nargs='*', help='Path discriminator hidden dim parameter', default=[512, 512])
    parser.add_argument('--disc_dropout_rate', type=float, help='Path discriminator dropout rate', default=0.3)

    # metadiscriminator parameters
    parser.add_argument('--metadisc_hidden', type=int, nargs='*', help='Meta discriminator hidden dim parameters', default=[512, 256])
    parser.add_argument('--metadisc_dropout_rate', type=float, help='Meta discriminator dropout rate', default=0.3)

    # AC model parameters
    parser.add_argument('--ac_hidden', type=int, nargs='*', help='ActorCritic hidden dim parameters', default=[512, 256])
    parser.add_argument('--actor_dropout_rate', type=float, help='actor dropout rate', default=0.3)
    parser.add_argument('--critic_dropout_rate', type=float, help='critic dropout rate', default=0.3)
    parser.add_argument('--act_dropout', type=float, help='action dropout rate', default=0.5)
    parser.add_argument('--target_update', type=float, help='update ratio of target network', default=0.05)
    parser.add_argument('--gamma', type=float, help='reward discount factor', default=0.99)

    # other training parameters
    parser.add_argument('--seed', type=int, help='Random seed (default: 1023)', default=1023)
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU to train model", default=False)
    parser.add_argument('--gpu', type=int, help='gpu device (default: 0)', default=0)

    # evaluation parameters
    parser.add_argument('--factor', type=float, help='decay factor for calculating path probability score (default: 0.9)', default=0.9)
    parser.add_argument('--topk', type=int, help='top ranked diseases to recommend', default=50)
    parser.add_argument('--eval_batch_size', type=int, help='batch size for evaluation step (default: 2)', default=2)
    parser.add_argument('--save_pred_paths', action="store_true", help="whether to save the predicted paths", default=False)

    args = parser.parse_args()

    if args.use_gpu and torch.cuda.is_available():
        use_gpu = True
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.set_device(args.gpu)
    elif args.use_gpu:
        print('No GPU is detected in this computer. Use CPU instead.')
        use_gpu = False
        device = 'cpu'
    else:
        use_gpu = False
        device = 'cpu'
    args.use_gpu = use_gpu
    args.device = device

    logger = utils.get_logger(os.path.join(args.log_dir,args.log_name))
    args.logger = logger
    logger.info(args)

    utils.set_random_seed(args.seed)

    ## create model folder
    folder_name = 'model_evaluation_results'
    if not os.path.isdir(os.path.join(args.output_folder, folder_name)):
        os.mkdir(os.path.join(args.output_folder, folder_name))
        args.pred_paths_save_location = os.path.join(args.output_folder, folder_name)
    else:
        args.pred_paths_save_location = os.path.join(args.output_folder, folder_name)

    ## find all disease ids
    type2id, id2type = utils.load_index(os.path.join(args.data_dir, 'type2freq.txt'))
    with open(os.path.join(args.data_dir, 'entity2typeid.pkl'), 'rb') as infile:
        entity2typeid = pickle.load(infile)
    disease_type = ['biolink:Disease', 'biolink:PhenotypicFeature', 'biolink:BehavioralFeature', 'biolink:DiseaseOrPhenotypicFeature']
    disease_type_ids = [type2id[x] for x in disease_type]
    args.disease_ids = [index for index, typeid in enumerate(entity2typeid) if typeid in disease_type_ids]

    ## start to evaluate ADAC model
    logger.info('Start to evaluate ADAC model')
    evaluate_model(args)