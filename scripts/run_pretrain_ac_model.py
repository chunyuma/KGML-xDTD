## This script is modified from the code used in the original ADAC RL model (Zhao et al. doi: 10.1145/3397271.3401171)
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
import copy
from tqdm import tqdm
from knowledge_graph import KnowledgeGraph
from kg_env import KGEnvironment
from models import PreActorCritic
from random import sample
from hummingbird.ml import convert

def pretrain(args):
    kg = KnowledgeGraph(args, bandwidth=args.bandwidth, entity_dim=args.entity_dim, entity_type_dim=args.entity_type_dim, relation_dim=args.relation_dim, emb_dropout_rate=args.emb_dropout_rate, bucket_interval=args.bucket_interval, load_graph=True)
    ## read pre-train model
    pretrain_model = joblib.load(args.pretrain_model_path)
    ## convert sklearn model to pytorch
    pretrain_model = convert(pretrain_model, 'pytorch')
    if args.use_gpu is True:
        pretrain_model.to(f"cuda:{args.gpu}")
    env = KGEnvironment(args, pretrain_model, kg, max_path_len=args.max_path, state_pre_history=args.state_history)
    pretrain_path_file = os.path.join(args.data_dir, args.expert_dir_name, args.path_file_name)
    with open(pretrain_path_file, 'rb') as infile:
        pretrain_path = pickle.load(infile)
    all_expert_path_size = pretrain_path[1].shape[0]
    logger.info("Load {} expert paths from {}".format(all_expert_path_size, pretrain_path_file))
    ## set up pretrained AC model
    model = PreActorCritic(args, kg, args.state_history, args.gamma, args.target_update, args.hidden)
    if args.use_pre_train:
        pretrain_ac_file = os.path.join(args.model_save_location, args.pre_train_model_name)
        logger.info("Load pretrained AC model from " + pretrain_ac_file)
        pretrain_dict = torch.load(pretrain_ac_file, map_location=args.device)
        ## load policy_net
        model_dict = model.policy_net.state_dict()
        model_dict.update(pretrain_dict)
        model.policy_net.load_state_dict(model_dict, strict=False)
        ## load target_net
        model.target_net.load_state_dict(model.policy_net.state_dict())
        model.target_net.eval()
        del pretrain_dict
        del model_dict
        if args.use_gpu:
            utils.empty_gpu_cache(args)

    logger.info('Policy Parameters:\n' + str(
        ([(i[0], 'requires_grad=' + str(i[1].requires_grad) + ' shape=' + str(i[1].shape)) for i in model.policy_net.named_parameters()])))
    logger.info('Target Parameters:\n' + str(
        ([(i[0], 'requires_grad=' + str(i[1].requires_grad) + ' shape=' + str(i[1].shape)) for i in model.target_net.named_parameters()])))
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.policy_net.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.scheduler_factor, patience=args.scheduler_patience)

    ### Pretrained AC ###
    model.policy_net.train()

    for epoch in range(1, args.epochs + 1):

        logger.info(f'running epoch{epoch}...')
        epoch_rewards, epoch_critic_loss, epoch_actor_loss = [], [], []
        ### Start epoch ###
        if all_expert_path_size > args.max_pre_path:
            new_indexes = np.random.permutation(all_expert_path_size)[:args.max_pre_path]
            new_pretrain_path = [pretrain_path[0][new_indexes],pretrain_path[1][new_indexes]]
        else:
            new_pretrain_path = pretrain_path
        dataloader = utils.ACDataLoader(list(range(new_pretrain_path[1].shape[0])), args.batch_size)
        total_steps = math.ceil(dataloader.num_paths/args.batch_size)
        dataloader.reset()
        step = 1
        # time_record = 0
        pbar = tqdm(total=dataloader.num_paths)
        while dataloader.has_next():
            # start_time = time.time()
            ### Start batch episodes ###
            episode_critic_loss = 0.0
            episode_actor_loss = 0.0
            episode_reward = 0.0

            batch_path_id = dataloader.get_batch()
            source_ids, target_ids = new_pretrain_path[1][batch_path_id][:,0], new_pretrain_path[1][batch_path_id][:,-1]
            env.initialize_path(source_ids, target_ids)
            act_num = 1
            while not env._done:
                batch_true_action = [new_pretrain_path[0][batch_path_id][:,act_num], new_pretrain_path[1][batch_path_id][:,act_num]]
                true_next_act = model.select_true_action(env._batch_curr_state, env._batch_curr_action_spaces, batch_true_action, args.device)
                env.batch_step(true_next_act)
                if args.use_gpu:
                    utils.empty_gpu_cache(args)
                update = 'actor' if epoch <= args.pre_actor_epoch else 'critic'
                critic_loss, actor_loss = model.imi_step_update(env._batch_curr_state, env._batch_curr_action_spaces, env._batch_curr_reward, env._done,
                                                                optimizer, args.device, update)
                if args.use_gpu:
                    utils.empty_gpu_cache(args)
                model.update_target_net_()
                if args.use_gpu:
                    utils.empty_gpu_cache(args)
                episode_critic_loss += critic_loss
                episode_actor_loss += actor_loss
                episode_reward += np.mean(env._batch_curr_reward)

                act_num += 1
            ### End of episodes ##

            env._batch_path = None
            env._batch_curr_action_spaces = None
            env._batch_curr_state = None
            env._batch_curr_reward = None

            epoch_rewards.append(episode_reward)
            epoch_critic_loss.append(episode_critic_loss)
            epoch_actor_loss.append(episode_actor_loss)
            env.reset()

            if args.use_gpu:
                utils.empty_gpu_cache(args)

            if step % 5 == 0:
                logger.info(f'epoch:{epoch:d} | step:{step:d}/{total_steps:d} | actor_loss_eps={episode_actor_loss:.5f} | critic_loss_eps={episode_critic_loss:.5f} | reward_eps={episode_reward:.5f}')

            step += 1
            pbar.update(len(source_ids))

        # Report performance
        avg_reward = np.mean(epoch_rewards)
        avg_critic_loss = np.mean(epoch_critic_loss)
        avg_actor_loss = np.mean(epoch_actor_loss)
        if update == 'actor':
            scheduler.step(avg_actor_loss)
        else:
            scheduler.step(avg_critic_loss)

        logger.info(f'epoch={epoch:d} | reward={avg_reward:.5f} | critic_loss={avg_critic_loss:.5f} | actor_loss={avg_actor_loss:.5f}')
        ### END of epoch ###

        # if epoch % args.save_every == 0 or epoch == args.epochs or epoch == args.pre_actor_epoch:
        if epoch % args.save_every == 0 or epoch == args.epochs or epoch == args.pre_actor_epoch:
            policy_file = os.path.join(args.model_save_location, f'pre_model_epoch{epoch}.pt') 
            logger.info("Save pre train model to " + policy_file)
            saved_sd = model.policy_net.state_dict()
            for para in model.policy_net.named_parameters():
                if para[1].requires_grad is False:
                    saved_sd.pop(para[0])
            torch.save(saved_sd, policy_file)
            saved_sd.clear()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## folder parameters
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default="../log_folder")
    parser.add_argument("--log_name", type=str, help="log file name", default="step10.log")
    parser.add_argument('--data_dir', type=str, help='Full path of data folder', default='../data')
    parser.add_argument('--path_file_name', type=str, default='expert_demonstration_relation_entity_max3.pkl', help='expert demonstration path file name')
    parser.add_argument('--text_emb_file_name', type=str, help='The name of text embedding file', default='embedding_biobert_namecat.pkl')
    parser.add_argument('--expert_dir_name', type=str, help='The name of expert path directory', default='expert_path_files')
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default="../models")

    ## knowledge graph and environment paramters
    parser.add_argument('--entity_dim', type=int, help='Dimension of entity embedding', default=100)
    parser.add_argument('--relation_dim', type=int, help='Dimension of relation embedding', default=100)
    parser.add_argument('--entity_type_dim', type=int, help='Dimension of entity type embedding', default=100)
    parser.add_argument('--max_path', type=int, help='Maximum length of path', default=3)
    parser.add_argument('--max_pre_path', type=int, help='Maximum number of pre-trained path', default=1000000)
    parser.add_argument('--bandwidth', type=int, help='Maximum number of neighbors', default=3000)
    parser.add_argument('--bucket_interval', type=int, help='adjacency list bucket size to save memory (default: 50)', default=50)
    parser.add_argument('--state_history', type=int, help='state history length', default=1)
    parser.add_argument("--emb_dropout_rate", type=float, help="Knowledge entity and relation embedding vector dropout rate (default: 0)", default=0)
    parser.add_argument("--pretrain_model_path", type=str, help="The path of pretrain model", default='../models/RF_model/RF_model.pt')
    parser.add_argument('--tp_reward', type=float, help='Reward if the agent hits the target entity (default: 1.0)', default=1.0)

    # model training parameters
    parser.add_argument('--seed', type=int, help='Random seed (default: 1023)', default=1023)
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU to train model", default=False)
    parser.add_argument('--gpu', type=int, help='gpu device (default: 0)', default=0)
    parser.add_argument('--batch_size', type=int, help='batch size (default: 256)', default=256)
    parser.add_argument('--epochs', type=int, help='Max number of epochs to pre train (default: 70)', default=70)
    parser.add_argument('--pre_actor_epoch', type=int, help='First pre train number of epochs for actor (default: 35)', default=35)
    parser.add_argument('--hidden', type=int, nargs='*', help='ActorCritic hidden dim parameters', default=[512, 256])
    parser.add_argument('--gamma', type=float, help='reward discount factor', default=0.99)
    parser.add_argument('--target_update', type=float, help='update ratio of target network', default=0.05)
    parser.add_argument('--actor_dropout_rate', type=float, help='actor dropout rate', default=0.3)
    parser.add_argument('--critic_dropout_rate', type=float, help='critic dropout rate', default=0.3)
    parser.add_argument('--lr', type=float, help='learning rate (default: 0.001)', default=0.001)
    parser.add_argument("--scheduler_patience", type=int, help="Number of epochs with no improvement after which learning rate will be reduced", default=5)
    parser.add_argument("--scheduler_factor", type=float, help="The factor for learning rate to be reduced", default=0.1)
    parser.add_argument('--save_every', type=int, help='Save the model every # epochs', default=1)
    parser.add_argument('--use_pre_train', action="store_true", help='use_pretrain_parameters', default=False)
    parser.add_argument('--pre_train_model_name', type=str, help='the name of certain epoch model')

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

    logger.info('Prepare data')

    ## create model folder
    folder_name = 'pretrain_AC_model'
    if not os.path.isdir(os.path.join(args.output_folder, folder_name)):
        os.mkdir(os.path.join(args.output_folder, folder_name))
        args.model_save_location = os.path.join(args.output_folder, folder_name)
    else:
        args.model_save_location = os.path.join(args.output_folder, folder_name)

    ## set up initial entity embedding, relation embedding
    kg_init_embedding_folder = os.path.join(args.data_dir, 'kg_init_embeddings')
    if not os.path.isdir(kg_init_embedding_folder):
        os.mkdir(kg_init_embedding_folder)
    
    if not os.path.isfile(os.path.join(kg_init_embedding_folder, 'entity_embeddings.npy')):
        with open(os.path.join(args.data_dir, 'text_embedding', args.text_emb_file_name),'rb') as infile:
            text_emb = pickle.load(infile)
        _, id2entity = utils.load_index(os.path.join(args.data_dir, 'entity2freq.txt'))
        entity_embeddings = np.array([text_emb[id2entity[key]] for key in id2entity if key != 0]).astype(float)
        np.save(os.path.join(kg_init_embedding_folder, 'entity_embeddings.npy'), entity_embeddings)

    if not os.path.isfile(os.path.join(kg_init_embedding_folder, 'relation_embeddings.npy')):
        _, id2relation = utils.load_index(os.path.join(args.data_dir, 'relation2freq.txt'))
        del id2relation[0]
        relation_embeddings = []
        for index in range(len(id2relation)):
            emb = [0] * len(id2relation)
            emb[index] = 1
            relation_embeddings += [emb]
        relation_embeddings = np.array(relation_embeddings).astype(float)
        np.save(os.path.join(kg_init_embedding_folder, 'relation_embeddings.npy'), relation_embeddings)

    ## find all disease ids
    type2id, id2type = utils.load_index(os.path.join(args.data_dir, 'type2freq.txt'))
    with open(os.path.join(args.data_dir, 'entity2typeid.pkl'), 'rb') as infile:
        entity2typeid = pickle.load(infile)
    disease_type = ['biolink:Disease', 'biolink:PhenotypicFeature', 'biolink:BehavioralFeature', 'biolink:DiseaseOrPhenotypicFeature']
    disease_type_ids = [type2id[x] for x in disease_type]
    args.disease_ids = [index for index, typeid in enumerate(entity2typeid) if typeid in disease_type_ids]

    ## start to pretrained AC model
    logger.info('Start to pretrained AC model')
    pretrain(args)