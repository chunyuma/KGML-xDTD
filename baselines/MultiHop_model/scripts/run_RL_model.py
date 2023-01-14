import argparse
import collections
import numpy as np
import pandas as pd
import os
import pickle
import utils
import time
import torch
import copy
import joblib
import random
from glob import glob
from tqdm import tqdm
from hummingbird.ml import convert
from knowledge_graph import KnowledgeGraph
from models import GraphSearchPolicy
from models import RewardShapingPolicyGradient

def run_model(args):

    ## set model path
    args.model_path = os.path.join(os.path.dirname(args.data_dir), 'models')

    ## construct RL model
    kg = KnowledgeGraph(args, load_graph=True)
    args.entity_dim = kg.entity_embeddings.weight.shape[1]
    args.relation_dim = kg.relation_embeddings.weight.shape[1]
    pn = GraphSearchPolicy(args)
    ## read pre-train model
    pretrain_model = joblib.load(args.pretrain_model_path)
    ## convert sklearn model to pytorch
    pretrain_model = convert(pretrain_model, 'pytorch')
    if use_gpu is True:
        pretrain_model.to(f"cuda:{args.gpu}")
    lf = RewardShapingPolicyGradient(args, kg, pn, pretrain_model)
    if use_gpu is True:
        lf.to(f"cuda:{args.gpu}")

    ## Read data
    logger.info('#### Read train and validation set ####')
    train_pairs = pd.read_csv(os.path.join(args.data_dir,'RL_model_train_val_test_data', 'train_pairs.txt'), sep='\t', header=0)
    train_pairs = train_pairs.apply(lambda row: [kg.entity2id[row[0]],kg.entity2id[row[1]]], axis=1, result_type='expand').to_numpy()
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

    ## Training RL model
    logger.info('#### Train RL model ####')
    lf.run_train(train_pairs, eval_drug_disease_dict, all_drug_disease_dict)
    logger.info('Model training is done!!!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default="~/explainable_DTD_MultiHopKG_model/log_folder")
    parser.add_argument("--log_name", type=str, help="log file name", default="step9.log")
    parser.add_argument("--data_dir", type=str, help="The path of data folder", default="~/explainable_DTD_MultiHopKG_model/data")
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default="~/explainable_DTD_MultiHopKG_model")

    ## knowledge graph and environment parameters
    parser.add_argument("--bandwidth", type=int, help="Maximum number of neighbor nodes to explore (default: 3000)", default=3000)
    parser.add_argument('--bucket_interval', type=int, default=50, help='adjacency list bucket size (default: 50)')
    parser.add_argument("--pretrain_model_path", type=str, help="The path of pretrain model", default='~/explainable_DTD_ADAC_model/models/RF_model/RF_model.pt')
    parser.add_argument('--tp_reward', type=float, help='Reward if the agent hits the target entity (default: 1.0)', default=1.0)
    parser.add_argument('--reward_shaping_threshold', type=float, help='Threshold cut off of reward shaping scores (default: 0.35)', default=0.35)

    # model parameters
    parser.add_argument('--use_init_emb', action="store_true", help="Use pre-train embedding as entity embedding and relation embedding", default=False)
    parser.add_argument("--entity_dim", type=int, help="The dimension of entity embedding vector (default: 100)", default=100)
    parser.add_argument("--relation_dim", type=int, help="The dimension of relation embedding vector (default: 100)", default=100)
    parser.add_argument("--lr", type=float, help="Model learning rate (default: 0.001)", default=0.001)
    parser.add_argument('--num_rollouts', type=int, help='number of rollouts (default: 20)', default=20)
    parser.add_argument('--num_rollout_steps', type=int, help='maximum path length (default: 3)', default=3)
    parser.add_argument('--history_num_layers', type=int, metavar='L', help='action history encoding LSTM number of layers (default: 3)', default=3)
    parser.add_argument('--history_dim', type=int, metavar='H', help='action history encoding LSTM hidden states dimension (default: 256)', default=256)
    parser.add_argument('--action_dropout_rate', type=float, help='Dropout rate for randomly masking out knowledge graph edges (default: 0.5)', default=0.5)
    parser.add_argument('--ff_dropout_rate', type=float, help='Feed-forward layer dropout rate (default: 0.3)', default=0.3)
    parser.add_argument('--gamma', type=float, help='reward discount factor (default: 0.99)', default=0.99)
    parser.add_argument('--beta', type=float, help='entropy regularization weight (default: 0.005)', default=0.005)
    parser.add_argument('--num_check_epochs', type=int, help='Number of epochs to check for stopping training when considering num_peek_epochs', default=20)
    parser.add_argument('--beam_size', type=int, help='size of beam used in beam search inference (default: 50)', default=50)
    parser.add_argument('--grad_norm', type=int, help='norm threshold for gradient clipping (default 0)', default=0)
    parser.add_argument("--emb_dropout_rate", type=float, help="Embedding vector dropout rate (default: 0.0)", default=0.0)
    parser.add_argument('--xavier_initialization', type=bool, help='Initialize all model parameters using xavier initialization (default: True)', default=True)

    # other training parameters
    parser.add_argument('--seed', type=int, help='Random seed (default: 1023)', default=1023)
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU to train model", default=False)
    parser.add_argument('--gpu', type=int, help='gpu device (default: 0)', default=0)
    parser.add_argument('--num_peek_epochs', type=int, help='Number of epochs to wait for next val set result check (default: 10)', default=10)
    parser.add_argument('--num_wait_epochs', type=int, help='Number of epochs to wait before stopping training if val set performance drops', default=500)
    parser.add_argument("--num_epochs", type=int, help="maximum number of pass over the entire training set (default: 1000)", default=1000)
    parser.add_argument("--train_batch_size", type=int, help="Train mini-batch size (default: 128)", default=128)
    parser.add_argument("--eval_batch_size", type=int, help="Evaluation (Val and Test) mini-batch size (default: 64)", default=64)
    parser.add_argument('--factor', type=float, help='decay factor for calculating path probability score (default: 0.9)', default=0.9)


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

    try:
        assert args.num_wait_epochs >= args.num_peek_epochs * args.num_check_epochs
    except AssertionError:
        logger.error(f"The parameter 'num_wait_epochs' has to be larger or equal to the multiplication of 'num_peek_epochs' and 'num_check_epochs'")
        exit()

    ## create output folder
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

    ## start to generate metapaths for train set for downstream training
    logger.info('training MultiHopKG model')
    run_model(args)

