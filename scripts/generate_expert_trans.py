import sys
import os
import argparse
import torch
import torch.nn as nn
import utils
import pickle
from models import ActorCritic, Transition

def gen_expert_trans(args):

    def _batch_get_state(batch_path, state_pre_history):
        path_len = batch_path[0].shape[1]
        return [batch_path[1][:,0], batch_path[0][:,max(0, path_len-state_pre_history-1):], batch_path[1][:,max(0, path_len-state_pre_history-1):]]

    # Generate expert actions and states
    pretrain_path_file = os.path.join(args.data_dir, args.expert_dir_name, args.path_file_name)
    with open(pretrain_path_file, 'rb') as infile:
        pretrain_path = pickle.load(infile)

    expert_transitions = []
    act_num = 1
    batch_path = [pretrain_path[0][:,:act_num], pretrain_path[1][:,:act_num]]

    # done = False
    while not (batch_path[0].shape[1] >= (args.max_path + 1)):
        batch_state = _batch_get_state(batch_path,args.state_history)
        expert_state_inputs = ActorCritic.process_state(args.state_history, batch_state)
        batch_expert_action = [pretrain_path[0][:,act_num], pretrain_path[1][:,act_num]]
        expert_transitions.append(Transition(expert_state_inputs, [], batch_expert_action, [], [], []))
        act_num += 1
        batch_path = [pretrain_path[0][:,:act_num], pretrain_path[1][:,:act_num]]

    expert_trans_file = os.path.join(args.data_dir, args.expert_dir_name, args.expert_trains_file_name)
    pickle.dump(expert_transitions, open(expert_trans_file, 'wb'))
    logger.info("Generate expert transitions to "+expert_trans_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## folder parameters
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default="../log_folder")
    parser.add_argument("--log_name", type=str, help="log file name", default="step9.log")
    parser.add_argument('--data_dir', type=str, help='Full path of data folder', default='../data')
    parser.add_argument('--path_file_name', type=str, default='expert_demonstration_relation_entity_max3.pkl', help='expert demonstration path file name')
    parser.add_argument('--expert_dir_name', type=str, help='The name of expert path directory', default='expert_path_files')
    parser.add_argument('--expert_trains_file_name', type=str, help='expert file name for output', default='expert_transitions.pkl')

    ## knowledge graph and environment paramters
    parser.add_argument('--max_path', type=int, help='Maximum length of path', default=3)
    parser.add_argument('--state_history', type=int, help='state history length', default=1)

    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir,args.log_name))
    args.logger = logger
    logger.info(args)

    ## start to generate transition information for expert paths
    logger.info('Start to generate transition information for expert paths')
    gen_expert_trans(args)
