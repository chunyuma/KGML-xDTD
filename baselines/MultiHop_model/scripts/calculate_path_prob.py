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
from glob import glob
import pandas as pd
import graph_tool.all as gt
from tqdm import tqdm, trange
from hummingbird.ml import convert
from knowledge_graph import KnowledgeGraph
from models import GraphSearchPolicy
from models import RewardShapingPolicyGradient
from node_synonymizer import NodeSynonymizer
nodesynonymizer = NodeSynonymizer()
import collections
import itertools
import copy

def load_page_rank_scores(kg, input_path):
    pgrk_scores = collections.defaultdict(float)
    with open(input_path) as f:
        for line in f:
            entity, score = line.strip().split('\t')
            entity_id = kg.entity2id[entity.strip()]
            score = float(score)
            pgrk_scores[entity_id] = score
    return pgrk_scores

def get_action_space(kg, source):
    action_space = []
    if source in kg.adj_list:
        for relation in kg.adj_list[source]:
            targets = kg.adj_list[source][relation]
            for target in targets:
                action_space.append((relation, target))
        if len(action_space) + 1 >= kg.bandwidth:
            sorted_action_space = sorted(action_space, key=lambda x: kg.page_rank_scores[x[1]], reverse=True)
            action_space = sorted_action_space[:kg.bandwidth]
    action_space.insert(0, (kg.self_edge, source))
    return action_space

def batch_get_true(args, batch_action_spaces, batch_true_actions):
    ((batch_r_space, batch_e_space), batch_action_mask) = batch_action_spaces
    if args.use_gpu:
        true_r = batch_true_actions[0].view(-1,1).cuda()
    else:
        true_r = batch_true_actions[0].view(-1,1)
    if args.use_gpu:
        true_e = batch_true_actions[1].view(-1,1).cuda()
    else:
        true_e = batch_true_actions[1].view(-1,1)
    true_idx_in_actor = torch.where((batch_r_space == true_r) * (batch_e_space == true_e))[1]

    return true_idx_in_actor, (true_r.view(-1), true_e.view(-1))

def select_true_action(args, action_space, action_dist, batch_true_actions):

    true_idx_in_actor, true_next_actions = batch_get_true(args, action_space, batch_true_actions)
    true_idx_in_actor = true_idx_in_actor.to(args.device)
    true_prob = action_dist.gather(1, true_idx_in_actor.view(-1, 1)).view(-1)
    weighted_logprob = torch.log((true_prob.view(-1,1)+utils.TINY_VALUE) * torch.count_nonzero(action_dist, dim=1).view(-1,1))

    return true_next_actions, weighted_logprob

def batch_calculate_prob_score(args, batch_paths, kg, model):

    model.eval()
    dataloader = utils.ACDataLoader(list(range(batch_paths[1].shape[0])), args.batch_size, permutation=False)

    # pbar = tqdm(total=dataloader.num_paths)
    pred_prob_scores = []
    while dataloader.has_next():

        batch_path_id = dataloader.get_batch()
        source_ids = batch_paths[1][batch_path_id][:,0].to(args.device)
        # Initialization
        if args.use_gpu:
            r_s = utils.int_fill_var_cuda(source_ids.size(), kg.self_edge, use_gpu=True)
        else:
            r_s = utils.int_fill_var_cuda(source_ids.size(), kg.self_edge, use_gpu=False)
        # if args.use_gpu:
        #     seen_nodes = utils.int_fill_var_cuda(source_ids.size(), kg.dummy_e, use_gpu=True).unsqueeze(1)
        # else:
        #     seen_nodes = utils.int_fill_var_cuda(source_ids.size(), kg.dummy_e, use_gpu=False).unsqueeze(1)
        seen_nodes = source_ids.unsqueeze(1)
        init_action = (r_s, source_ids)
        model.pn.initialize_path(init_action, kg)

        if args.use_gpu:
            action_log_weighted_prob = utils.zeros_var_cuda(len(batch_path_id), use_gpu=True)
        else:
            action_log_weighted_prob = utils.zeros_var_cuda(len(batch_path_id), use_gpu=False)

        action = init_action
        print(f"", flush=True)
        for t in range(args.max_path):
            print(f"Here is step {t+1} for beam search", flush=True)
            last_r, e = action
            obs = [source_ids, t==(args.max_path-1), last_r, seen_nodes]
            batch_true_actions = [batch_paths[0][batch_path_id][:,t+1], batch_paths[1][batch_path_id][:,t+1]]
            # one step forward in search
            db_outcomes, _, _ = model.pn.transit(e, obs, kg, merge_aspace_batching_outcome=True)
            action_space, action_dist = db_outcomes[0]
            action, weighted_logprob = select_true_action(args, action_space, action_dist, batch_true_actions)
            action_log_weighted_prob = action_log_weighted_prob.view(-1, 1) + args.factor**t * weighted_logprob
            model.pn.update_path(action, kg)
            if args.use_gpu:
                seen_nodes = torch.cat([seen_nodes, action[1].unsqueeze(1)], dim=1)
            else:
                seen_nodes = torch.cat([seen_nodes, action[1].unsqueeze(1).cpu()], dim=1)
            torch.cuda.empty_cache()

        pred_prob_scores += [action_log_weighted_prob.view(-1).cpu().detach()]

        if args.use_gpu:
            torch.cuda.empty_cache()
        # pbar.update(len(source_ids))

    return np.concatenate(pred_prob_scores)

def translate_path_to_ids(path, kg):
	return [kg.entity2id[x] for index, x in enumerate(path) if index%2==0]

def evaluate_path(args):
    kg = KnowledgeGraph(args, load_graph=True)
    args.entity_dim = kg.entity_embeddings.weight.shape[1]
    args.relation_dim = kg.relation_embeddings.weight.shape[1]
    pn = GraphSearchPolicy(args)
    ## read pre-train model
    pretrain_model = joblib.load(args.pretrain_model_path)
    ## convert sklearn model to pytorch
    pretrain_model = convert(pretrain_model, 'pytorch')
    if args.use_gpu is True:
        pretrain_model.to(f"cuda:{args.gpu}")
    lf = RewardShapingPolicyGradient(args, kg, pn, pretrain_model)
    if len(glob(os.path.join(args.model_path,'*_model_best.tar'))) == 1:
        best_model_path = glob(os.path.join(args.model_path,'*_model_best.tar'))[0]
    else:
        args.logger.error(f"Can't find the best model from {args.model_path}")
        exit()
    lf.load_checkpoint(best_model_path)
    if args.use_gpu is True:
        lf.to(f"cuda:{args.gpu}")
    lf.eval()

    with open(args.target_paths_file, 'rb') as infile:
        paths_temp = pickle.load(infile)
    pbar = tqdm(total=len(paths_temp))
    for (source, target) in paths_temp:
        batch_paths = paths_temp[(source, target)]
        if len(batch_paths[1]) == 0:
            continue
        pred_prob_scores = batch_calculate_prob_score(args, batch_paths, kg, lf)
        pred_prob_scores = torch.tensor(pred_prob_scores)
        paths_temp[(source, target)] = [batch_paths[0], batch_paths[1], pred_prob_scores]
        pbar.update(1)

    with open(args.output_file, 'wb') as outfile:
        pickle.dump(paths_temp, outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## folder parameters
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default="~/explainable_DTD_MultiHopKG_model/log_folder")
    parser.add_argument("--log_name", type=str, help="log file name", default="step9.log")
    parser.add_argument("--data_dir", type=str, help="The path of data folder", default="~/explainable_DTD_MultiHopKG_model/data")
    parser.add_argument('--target_paths_file', type=str, help='The full path of target path file')
    parser.add_argument('--output_file', type=str, help='The full path of output file')

    ## knowledge graph and environment parameters
    parser.add_argument('--max_path', type=int, help='Maximum length of path', default=3)
    parser.add_argument("--bandwidth", type=int, help="Maximum number of neighbor nodes to explore (default: 3000)", default=3000)
    parser.add_argument('--bucket_interval', type=int, default=50, help='adjacency list bucket size (default: 50)')
    parser.add_argument("--pretrain_model_path", type=str, help="The path of pretrain model", default='~/explainable_DTD_ADAC_model/models/RF_model/RF_model.pt')
    parser.add_argument('--tp_reward', type=float, help='Reward if the agent hits the target entity (default: 1.0)', default=1.0)
    parser.add_argument('--reward_shaping_threshold', type=float, help='Threshold cut off of reward shaping scores (default: 0.35)', default=0.35)

    # model parameters
    parser.add_argument('--use_init_emb', action="store_true", help="Use pre-train embedding as entity embedding and relation embedding", default=False)
    parser.add_argument("--entity_dim", type=int, help="The dimension of entity embedding vector (default: 100)", default=100)
    parser.add_argument("--relation_dim", type=int, help="The dimension of relation embedding vector (default: 100)", default=100)
    parser.add_argument('--num_rollouts', type=int, help='number of rollouts (default: 20)', default=20)
    parser.add_argument('--num_rollout_steps', type=int, help='maximum path length (default: 3)', default=3)
    parser.add_argument('--history_num_layers', type=int, metavar='L', help='action history encoding LSTM number of layers (default: 3)', default=3)
    parser.add_argument('--history_dim', type=int, metavar='H', help='action history encoding LSTM hidden states dimension (default: 200)', default=200)
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

    # additional parameters
    parser.add_argument('--factor', type=float, help='decay factor for calculating path probability score (default: 0.9)', default=0.9)
    parser.add_argument('--batch_size', type=int, help='batch size (default:2000)', default=2000)
    parser.add_argument('--eval_batch_size', type=int, help='Evaluation (Val and Test) mini-batch size (default: 64)', default=64)
    parser.add_argument('--topk', type=int, help='checking top K paths', default=50)

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

    ## find all disease ids
    type2id, id2type = utils.load_index(os.path.join(args.data_dir, 'type2freq.txt'))
    with open(os.path.join(args.data_dir, 'entity2typeid.pkl'), 'rb') as infile:
        entity2typeid = pickle.load(infile)
    disease_type = ['biolink:Disease', 'biolink:PhenotypicFeature', 'biolink:BehavioralFeature', 'biolink:DiseaseOrPhenotypicFeature']
    disease_type_ids = [type2id[x] for x in disease_type]
    args.disease_ids = [index for index, typeid in enumerate(entity2typeid) if typeid in disease_type_ids]

    ## set model path
    args.model_path = os.path.join(os.path.dirname(args.data_dir), 'models')

    ## add fake parameters
    args.train_batch_size = args.batch_size
    args.num_epochs = 1
    args.num_wait_epochs = 1
    args.num_peek_epochs = 1
    args.num_check_epochs = 1
    args.lr = 0.0005

    ## start to evaluate the predicted paths
    logger.info('Start to evaluate the predicted paths')
    evaluate_path(args)
