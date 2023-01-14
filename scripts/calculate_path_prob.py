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
import graph_tool.all as gt
from tqdm import tqdm, trange
from knowledge_graph import KnowledgeGraph
from kg_env import KGEnvironment
from models import DiscriminatorActorCritic
from random import sample, choices
from hummingbird.ml import convert
from node_synonymizer import NodeSynonymizer
nodesynonymizer = NodeSynonymizer()
import collections
import itertools

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
        true_r = batch_true_actions[0].view(-1,1).to(args.device)
    else:
        true_r = batch_true_actions[0].view(-1,1)
    if args.use_gpu:
        true_e = batch_true_actions[1].view(-1,1).to(args.device)
    else:
        true_e = batch_true_actions[1].view(-1,1)
    true_idx_in_actor = torch.where((batch_r_space == true_r) * (batch_e_space == true_e))[1]

    return true_idx_in_actor, (true_r, true_e)

def select_true_action(model, batch_state, batch_action_spaces, batch_true_actions, args):
    device = args.device
    state_inputs = model.process_state(model.history_len, batch_state).to(device)
    true_idx_in_actor, true_next_actions = batch_get_true(args, batch_action_spaces, batch_true_actions)

    probs, _ = model.policy_net(state_inputs, batch_action_spaces)
    if args.use_gpu:
        utils.empty_gpu_cache(args)
    true_idx_in_actor = true_idx_in_actor.to(device)
    true_prob = probs.gather(1, true_idx_in_actor.view(-1, 1)).view(-1)
    weighted_logprob = torch.log((true_prob.view(-1,1)+utils.TINY_VALUE) * torch.count_nonzero(probs, dim=1).view(-1,1))

    return true_next_actions, weighted_logprob

def batch_calculate_prob_score(args, batch_paths, env, model):

    env.reset()
    model.policy_net.eval()
    dataloader = utils.ACDataLoader(list(range(batch_paths[1].shape[0])), args.batch_size, permutation=False)

    # pbar = tqdm(total=dataloader.num_paths)
    pred_prob_scores = []
    while dataloader.has_next():

        batch_path_id = dataloader.get_batch()
        source_ids = batch_paths[1][batch_path_id][:,0]
        env.initialize_path(source_ids)
        act_num = 1

        if args.use_gpu:
            action_log_weighted_prob = utils.zeros_var_cuda(len(batch_path_id), args=args, use_gpu=True)
        else:
            action_log_weighted_prob = utils.zeros_var_cuda(len(batch_path_id), args=args, use_gpu=False)

        while not env._done:
            batch_true_action = [batch_paths[0][batch_path_id][:,act_num], batch_paths[1][batch_path_id][:,act_num]]
            true_next_act, weighted_logprob = select_true_action(model, env._batch_curr_state, env._batch_curr_action_spaces, batch_true_action, args)
            env.batch_step(true_next_act)
            if args.use_gpu:
                utils.empty_gpu_cache(args)
            action_log_weighted_prob = action_log_weighted_prob.view(-1, 1) + args.factor**(act_num-1) * weighted_logprob
            if args.use_gpu:
                utils.empty_gpu_cache(args)
            act_num += 1
        ### End of episodes ##

        pred_prob_scores += [action_log_weighted_prob.view(-1).cpu().detach()]
        env.reset()

        if args.use_gpu:
            utils.empty_gpu_cache(args)
        # pbar.update(len(source_ids))

    return np.concatenate(pred_prob_scores)

def translate_path_to_ids(path, kg):
	return [kg.entity2id[x] for index, x in enumerate(path) if index%2==0]

def evaluate_path(args):
    kg = KnowledgeGraph(args, bandwidth=args.bandwidth, entity_dim=args.entity_dim, entity_type_dim=args.entity_type_dim, relation_dim=args.relation_dim, emb_dropout_rate=args.emb_dropout_rate, bucket_interval=args.bucket_interval, load_graph=True)
    ## read pre-train model
    pretrain_model = joblib.load(args.pretrain_model_path)
    ## convert sklearn model to pytorch
    pretrain_model = convert(pretrain_model, 'pytorch')
    if args.use_gpu is True:
        pretrain_model.to(f"cuda:{args.gpu}")
    env = KGEnvironment(args, pretrain_model, kg, max_path_len=args.max_path, state_pre_history=args.state_history)
    ## load the best model
    model = DiscriminatorActorCritic(args, kg, args.state_history, args.gamma, args.target_update, args.ac_hidden, args.disc_hidden, args.metadisc_hidden)
    args.logger.info(f"Loading the best model from {args.policy_net_file}")
    policy_net = torch.load(args.policy_net_file, map_location=args.device)
    model_temp = model.policy_net.state_dict()
    model_temp.update(policy_net)
    model.policy_net.load_state_dict(model_temp)
    del policy_net
    del model_temp

    with open(args.target_paths_file, 'rb') as infile:
        paths_temp = pickle.load(infile)
    pbar = tqdm(total=len(paths_temp))
    for (source, target) in paths_temp:
        batch_paths = paths_temp[(source, target)]
        if len(batch_paths[1]) == 0:
            continue
        pred_prob_scores = batch_calculate_prob_score(args, batch_paths, env, model)
        pred_prob_scores = torch.tensor(pred_prob_scores)
        paths_temp[(source, target)] = [batch_paths[0], batch_paths[1], pred_prob_scores]
        pbar.update(1)

    with open(args.output_file, 'wb') as outfile:
        pickle.dump(paths_temp, outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## folder parameters
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default="../log_folder")
    parser.add_argument("--log_name", type=str, help="log file name", default="step15_2.log")
    parser.add_argument('--data_dir', type=str, help='Full path of data folder', default='../data')
    parser.add_argument('--policy_net_file', type=str, help='The full path of trained policy network model')
    parser.add_argument('--target_paths_file', type=str, help='The full path of target path file')
    parser.add_argument('--output_file', type=str, help='The full path of output file')

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

    # additional parameters
    parser.add_argument('--factor', type=float, help='decay factor for calculating path probability score (default: 0.9)', default=0.9)
    parser.add_argument('--batch_size', type=int, help='batch size (default: 800)', default=800)

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

    ## start to evaluate the predicted paths
    logger.info('Start to evaluate the predicted paths')
    evaluate_path(args)
