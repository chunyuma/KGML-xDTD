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
import pandas as pd
from tqdm import tqdm
from knowledge_graph import KnowledgeGraph
from kg_env import KGEnvironment
from models import DiscriminatorActorCritic, Transition
from random import sample, choices, shuffle
from hummingbird.ml import convert

def train(args):
    kg = KnowledgeGraph(args, bandwidth=args.bandwidth, entity_dim=args.entity_dim, entity_type_dim=args.entity_type_dim, relation_dim=args.relation_dim, emb_dropout_rate=args.emb_dropout_rate, bucket_interval=args.bucket_interval, load_graph=True)
    ## read pre-train model
    pretrain_model = joblib.load(args.pretrain_model_path)
    ## convert sklearn model to pytorch
    pretrain_model = convert(pretrain_model, 'pytorch')
    if args.use_gpu is True:
        pretrain_model.to(f"cuda:{args.gpu}")
    env = KGEnvironment(args, pretrain_model, kg, max_path_len=args.max_path, state_pre_history=args.state_history)
    ## set up ADAC model
    model = DiscriminatorActorCritic(args, kg, args.state_history, args.gamma, args.target_update, args.ac_hidden, args.disc_hidden, args.metadisc_hidden)
    logger.info('Discriminator Parameters:\n' + str(
        ([(i[0], 'requires_grad=' + str(i[1].requires_grad) + ' shape=' + str(i[1].shape)) for i in model.discriminator.named_parameters()])))
    logger.info('MetaDiscriminator Parameters:\n' + str(
        ([(i[0], 'requires_grad=' + str(i[1].requires_grad) + ' shape=' + str(i[1].shape)) for i in model.metadiscriminator.named_parameters()])))
    logger.info('Policy Parameters:\n' + str(
        ([(i[0], 'requires_grad=' + str(i[1].requires_grad) + ' shape=' + str(i[1].shape)) for i in model.policy_net.named_parameters()])))
    disc_optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.discriminator.parameters()), lr=args.disc_lr)
    metadisc_optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.metadiscriminator.parameters()), lr=args.metadisc_lr)
    ac_optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.policy_net.parameters()), lr=args.ac_lr)

    ## read expert path transition state
    expert_path_trains_file = os.path.join(args.data_dir, args.expert_dir_name, args.path_trans_file_name)
    with open(expert_path_trains_file, 'rb') as infile:
        expert_transitions = pickle.load(infile)
    expert_trans_size = expert_transitions[0].state.shape[0]
    logger.info("Load {} expert transitions from {}".format(expert_trans_size, expert_path_trains_file))
    expert_transitions_source_df = pd.DataFrame(expert_transitions[0].state[:,0].numpy())
    expert_transitions_check_list = [torch.hstack([expert_transitions[i].state,torch.vstack(expert_transitions[i].action).T]).unique(dim=0) for i in range(len(expert_transitions))]
    expert_transitions_check_dict = {tuple(x):True for step in expert_transitions_check_list for x in step.tolist()}
    del expert_transitions_check_list

    ## read expert path
    expert_path_file = os.path.join(args.data_dir, args.expert_dir_name, args.path_file_name)
    with open(expert_path_file, 'rb') as infile:
        expert_path = pickle.load(infile)
    all_expert_path_size = expert_path[1].shape[0]
    logger.info("Load {} expert paths from {}".format(all_expert_path_size, expert_path_file))

    # only consider the entity meta path
    expert_metapath_temp = expert_path[1].clone()[:,1:]
    expert_metapath_temp.apply_(lambda x: kg.entity2typeid[x])
    expert_metapath = expert_metapath_temp.unique(dim=0).clone()

    if args.warmup:
        # load pretrained AC model parameters
        pretrain_ac_file = os.path.join(args.output_folder, 'pretrain_AC_model', args.pre_ac_file)
        logger.info("Load pretrained AC model from " + pretrain_ac_file)
        pretrain_dict = torch.load(pretrain_ac_file, map_location=args.device)
        model_dict = model.policy_net.state_dict()
        model_dict.update(pretrain_dict)
        model.policy_net.load_state_dict(model_dict)
        del pretrain_dict
        del model_dict
        if args.use_gpu:
            utils.empty_gpu_cache(args)


    ## read train dataset for ADAC model
    train_pairs = pd.read_csv(os.path.join(args.data_dir,'RL_model_train_val_test_data', 'train_pairs.txt'), sep='\t', header=0)
    train_pairs = train_pairs.apply(lambda row: [kg.entity2id[row[0]],kg.entity2id[row[1]]], axis=1, result_type='expand').to_numpy()
    if args.train_batch_size%args.num_rollouts != 0:
        logger.warning(f'train_batch_size is not the multiple of num_rollouts, thus the batch size is automatically changed to {args.train_batch_size//args.num_rollouts * args.num_rollouts}')
        args.train_batch_size = args.train_batch_size//args.num_rollouts * args.num_rollouts

    train_dataloader = utils.ACDataLoader(list(range(train_pairs.shape[0])), args.train_batch_size//args.num_rollouts)  

    model.policy_net.train()
    model.discriminator.train()
    model.metadiscriminator.train()
    total_steps = math.ceil(train_dataloader.num_paths/(args.train_batch_size//args.num_rollouts))

    # for epoch in range(1, args.epochs + 1):
    for epoch in range(1, args.epochs + 1):

        logger.info(f'running epoch{epoch}...')

        epoch_rewards, epoch_critic_loss, epoch_entropy_loss, epoch_actor_loss, epoch_disc_loss, epoch_metadisc_loss, epoch_hit = [], [], [], [], [], [], []
        train_dataloader.reset()
        step = 1
        pbar = tqdm(total=train_dataloader.num_paths)
        while train_dataloader.has_next():

            transitions = []
            outputs = []
            batch_path_id = train_dataloader.get_batch()
            source_ids, target_ids = np.repeat(train_pairs[batch_path_id][:,0], args.num_rollouts), np.repeat(train_pairs[batch_path_id][:,1], args.num_rollouts)
            env.initialize_path(source_ids, target_ids)

            batch_state_inputs = DiscriminatorActorCritic.process_state(model.history_len, env._batch_curr_state)
            batch_action_spaces = env._batch_curr_action_spaces

            # policy rollout
            rollout_timesteps = 0
            batch_path = None
            episode_reward = 0.0
            episode_hit = 0
            while not env._done:
                rollout_timesteps += 1
                batch_next_act, batch_saved_output = model.select_action(batch_state_inputs, batch_action_spaces, args.device, args.act_dropout)
                env.batch_step(batch_next_act)
                # batch_action = [env._batch_path[0][:,-1],env._batch_path[1][:,-1]]
                if not env._done:
                    batch_next_state_inputs = model.process_state(model.history_len, env._batch_curr_state)
                    batch_next_action_spaces = env._batch_curr_action_spaces
                else:
                    batch_next_state_inputs, batch_next_action_spaces = None, None
                    batch_path = env._batch_path
                transitions.append(
                    Transition(
                        batch_state_inputs,
                        batch_action_spaces,
                        [env._batch_path[0][:,-1], env._batch_path[1][:,-1]],
                        env._batch_curr_reward,
                        batch_next_state_inputs,
                        batch_next_action_spaces
                    )
                )
                episode_reward += np.mean(env._batch_curr_reward)
                outputs.append(batch_saved_output)
                batch_state_inputs, batch_action_spaces = batch_next_state_inputs, batch_next_action_spaces
                if args.use_gpu:
                    utils.empty_gpu_cache(args)
            episode_hit = (batch_path[1][:,-1] == torch.tensor(target_ids)).float().mean().item()
            ### End of episodes ##

            # update path discriminator
            episode_disc_loss_list = []
            expert_trans_curr_idx = list(expert_transitions_source_df.loc[expert_transitions_source_df[0].isin(set(source_ids)),:].index)
            for _ in range(args.disc_per_step):
                shuffle(expert_trans_curr_idx)
                if len(expert_trans_curr_idx) > len(source_ids):
                    sampled_idx = sample(expert_trans_curr_idx, k=len(source_ids))
                else:
                    sampled_idx = choices(expert_trans_curr_idx, k=len(source_ids))
                expert_batch = [utils.get_expert_trans(transition, sampled_idx) for transition in expert_transitions]
                # label = np.zeros_like(transitions[-1].reward)
                for i in range(rollout_timesteps):
                    expert_trans = expert_batch[i]
                    train_trans = transitions[i]
                    train_trans_temp = torch.hstack([train_trans.state,torch.vstack(train_trans.action).T])
                    label = [expert_transitions_check_dict.get(tuple(state_temp.tolist()), False) for state_temp in train_trans_temp]
                    disc_loss_temp = model.step_update_discriminator(expert_trans, train_trans, label, disc_optimizer, args.device)
                    episode_disc_loss_list.append(disc_loss_temp)

                if args.use_gpu:
                    utils.empty_gpu_cache(args)
            episode_disc_loss = np.mean(episode_disc_loss_list)

            # update meta-path discriminator
            batch_metapath_temp = batch_path[1].clone()[:,1:]
            batch_metapath_temp.apply_(lambda x: kg.entity2typeid[x])
            batch_metapath = batch_metapath_temp.clone()
            meta_label = [(pattern == expert_metapath).all(dim=1).numpy().sum() > 0 for pattern in batch_metapath]
            episode_metadisc_loss_list = []
            for _ in range(args.metadisc_per_step):
                if len(expert_metapath) > len(source_ids):
                    batch_expert_metapath = expert_metapath[sample(range(len(expert_metapath)), k=len(source_ids))]
                else:
                    batch_expert_metapath = expert_metapath[choices(range(len(expert_metapath)), k=len(source_ids))]
                episode_metadisc_loss_temp = model.update_metadiscriminator(batch_expert_metapath, batch_metapath, meta_label, metadisc_optimizer, args.device)
                episode_metadisc_loss_list.append(episode_metadisc_loss_temp)

                if args.use_gpu:
                    utils.empty_gpu_cache(args)
            episode_metadisc_loss = np.mean(episode_metadisc_loss_list)

            # update AC model
            episode_critic_loss = 0.0
            episode_actor_loss = 0.0
            episode_entropy_loss = 0.0
            ac_reward = [trans.reward for trans in transitions]
            batch_rewards = np.vstack(ac_reward).T
            batch_rewards = torch.tensor(batch_rewards).to(args.device)
            num_steps = batch_rewards.shape[1]
            for i in range(1, num_steps):
                batch_rewards[:, num_steps - i - 1] += args.gamma * batch_rewards[:, num_steps - i]

            if epoch > args.ac_update_delay:
                actor_loss_list = []
                critic_loss_list = []
                entropy_loss_list = []
                for i in range(rollout_timesteps):
                    train_trans = transitions[i]
                    saved_output = outputs[i]
                    batch_reward = batch_rewards[:, i]
                    # last_step = i == rollout_timesteps -1
                    last_step = False
                    actor_loss, critic_loss, entropy_loss = model.step_update_with_expert(
                        batch_metapath, saved_output, train_trans, batch_reward, args.device, args.disc_alpha, args.metadisc_alpha, last_step
                    )

                    actor_loss_list += [actor_loss]
                    critic_loss_list += [critic_loss]
                    entropy_loss_list += [entropy_loss]
                    if args.use_gpu:
                        utils.empty_gpu_cache(args)

                ep_avg_actor_loss = torch.mean(torch.stack(actor_loss_list))
                episode_actor_loss = ep_avg_actor_loss.item()
                ep_avg_critic_loss = torch.mean(torch.stack(critic_loss_list))
                episode_critic_loss = ep_avg_critic_loss.item()
                ep_avg_entropy_loss = torch.mean(torch.stack(entropy_loss_list))
                episode_entropy_loss = ep_avg_entropy_loss.item()
                ac_optimizer.zero_grad()
                loss = ep_avg_actor_loss + ep_avg_critic_loss + args.ent_weight * ep_avg_entropy_loss
                loss.backward()
                ac_optimizer.step()

            del transitions
            del outputs
            env.reset()

            if args.use_gpu:
                utils.empty_gpu_cache(args)

            epoch_rewards.append(episode_reward)
            epoch_critic_loss.append(episode_critic_loss)
            epoch_entropy_loss.append(episode_entropy_loss)
            epoch_actor_loss.append(episode_actor_loss)
            epoch_disc_loss.append(episode_disc_loss)
            epoch_metadisc_loss.append(episode_metadisc_loss)
            epoch_hit.append(episode_hit)
            if step % args.every_print == 0:
                logger.info(f'epoch: {epoch:d} | step:{step:d}/{total_steps:d} | actor_loss_eps={episode_actor_loss:.5f} | critic_loss_eps={episode_critic_loss:.5f} | disc_loss_eps={episode_disc_loss:.5f} | metadisc_loss_eps={episode_metadisc_loss:.5f} | entropy_loss_eps={episode_entropy_loss:.5f} | reward_eps={episode_reward:.5f} | sucessful_hits_eps={episode_hit:.5f}')
            
            step += 1
            pbar.update(len(batch_path_id))

        # Report performance
        avg_reward = np.mean(epoch_rewards)
        avg_critic_loss = np.mean(epoch_critic_loss)
        avg_actor_loss = np.mean(epoch_actor_loss)
        avg_disc_loss = np.mean(epoch_disc_loss)
        avg_entropy_loss = np.mean(epoch_entropy_loss)
        avg_metadisc_loss = np.mean(epoch_metadisc_loss)
        avg_hit = np.mean(epoch_hit)
        logger.info(f'epoch={epoch:d} | avg actor loss={avg_actor_loss:.5f} | avg critic loss={avg_critic_loss:.5f} | disc_loss={avg_disc_loss:.5f} | metadisc_loss={avg_metadisc_loss:.5f} | entropy_loss={avg_entropy_loss:.5f} | avg reward={avg_reward:.5f} | average sucessful hits={avg_hit:.5f}')
        ### END of epoch ###

        ## save the best model
        if epoch > args.ac_update_delay:
            if not os.path.isdir(os.path.join(args.model_save_location, 'policy_net')):
                os.mkdir(os.path.join(args.model_save_location, 'policy_net'))
                policy_model_save_location = os.path.join(args.model_save_location, 'policy_net')
            else:
                policy_model_save_location = os.path.join(args.model_save_location, 'policy_net')
            policy_file = os.path.join(policy_model_save_location, f'policy_model_epoch{epoch}.pt') 
            policy_saved_sd = model.policy_net.state_dict()
            for para in model.policy_net.named_parameters():
                if para[1].requires_grad is False:
                    policy_saved_sd.pop(para[0])
            torch.save(policy_saved_sd, policy_file)
            policy_saved_sd.clear()
            logger.info("Save policy to " + policy_file)

            if not os.path.isdir(os.path.join(args.model_save_location, 'disc_net')):
                os.mkdir(os.path.join(args.model_save_location, 'disc_net'))
                disc_model_save_location = os.path.join(args.model_save_location, 'disc_net')
            else:
                disc_model_save_location = os.path.join(args.model_save_location, 'disc_net')
            disc_file = os.path.join(disc_model_save_location, f'discriminator_model_epoch{epoch}.pt')
            disc_saved_sd = model.discriminator.state_dict()
            for para in model.discriminator.named_parameters():
                if para[1].requires_grad is False:
                    disc_saved_sd.pop(para[0])
            torch.save(disc_saved_sd, disc_file)
            disc_saved_sd.clear()
            logger.info("Save discriminator to " + disc_file)

            if not os.path.isdir(os.path.join(args.model_save_location, 'metadisc_net')):
                os.mkdir(os.path.join(args.model_save_location, 'metadisc_net'))
                metadisc_model_save_location = os.path.join(args.model_save_location, 'metadisc_net')
            else:
                metadisc_model_save_location = os.path.join(args.model_save_location, 'metadisc_net')
            metadisc_file = os.path.join(metadisc_model_save_location, f'metadiscriminator_model_epoch{epoch}.pt')
            metadisc_saved_sd = model.metadiscriminator.state_dict()
            for para in model.metadiscriminator.named_parameters():
                if para[1].requires_grad is False:
                    metadisc_saved_sd.pop(para[0])
            torch.save(metadisc_saved_sd, metadisc_file)
            metadisc_saved_sd.clear()
            logger.info("Save metadiscriminator to " + metadisc_file)


        # # Evaluate model with evaluation data set
        # if args.run_eval and (epoch % args.save_every == 0 or epoch == args.epochs):
        #     env.reset()
        #     results = utils.evaluate(args, val_drug_disease_dict, env, model, all_drug_disease_dict)
        #     if args.use_gpu:
        #         utils.empty_gpu_cache(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## folder parameters
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default="../log_folder")
    parser.add_argument("--log_name", type=str, help="log file name", default="step11.log")
    parser.add_argument('--data_dir', type=str, help='Full path of data folder', default='../data')
    parser.add_argument('--path_file_name', type=str, default='expert_demonstration_relation_entity_max3.pkl', help='expert demonstration path file name')
    parser.add_argument('--path_trans_file_name', type=str, default='expert_transitions.pkl', help='expert transitions path file name')
    parser.add_argument('--text_emb_file_name', type=str, help='The name of text embedding file', default='embedding_biobert_namecat.pkl')
    parser.add_argument('--expert_dir_name', type=str, help='The name of expert path directory', default='expert_path_files')
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default="../models")

    ## knowledge graph and environment parameters
    parser.add_argument('--entity_dim', type=int, help='Dimension of entity embedding', default=100)
    parser.add_argument('--relation_dim', type=int, help='Dimension of relation embedding', default=100)
    parser.add_argument('--entity_type_dim', type=int, help='Dimension of entity type embedding', default=100)
    parser.add_argument('--max_path', type=int, help='Maximum length of path', default=3)
    parser.add_argument('--bandwidth', type=int, help='Maximum number of neighbors', default=3000)
    parser.add_argument('--bucket_interval', type=int, help='adjacency list bucket size to save memory (default: 50)', default=50)
    parser.add_argument('--state_history', type=int, help='state history length', default=1)
    parser.add_argument('--num_rollouts', type=int, help='number of rollouts (default: 16)', default=16)
    parser.add_argument("--emb_dropout_rate", type=float, help="Knowledge entity and relation embedding vector dropout rate (default: 0)", default=0)
    parser.add_argument("--pretrain_model_path", type=str, help="The path of pretrain model", default='../models/RF_model/RF_model.pt')
    parser.add_argument('--tp_reward', type=float, help='Reward if the agent hits the target entity (default: 1.0)', default=1.0)

    # pretrained AC model parameters
    parser.add_argument('--warmup', action="store_true", help='whether to use pre train AC model', default=False)
    parser.add_argument('--pre_ac_file', type=str, help='pre train file name', default='pre_model_epoch70.pt')

    # discriminator parameters
    parser.add_argument('--disc_hidden', type=int, nargs='*', help='Path discriminator hidden dim parameter', default=[512, 512])
    parser.add_argument('--disc_lr', type=float, help='learning rate for path discriminator', default=0.0005)
    parser.add_argument('--disc_per_step', type=int, help='Number of path discriminator updates for every step', default=20)
    parser.add_argument('--disc_alpha', type=float, help='factor for expert reward', default=0.006)
    parser.add_argument('--disc_dropout_rate', type=float, help='Path discriminator dropout rate', default=0.3)

    # metadiscriminator parameters
    parser.add_argument('--metadisc_hidden', type=int, nargs='*', help='Meta discriminator hidden dim parameters', default=[512, 256])
    parser.add_argument('--metadisc_lr', type=float, help='learning rate for metadiscriminator', default=0.0005)
    parser.add_argument('--metadisc_per_step', type=int, help='Number of metadiscriminator updates for every step', default=1)
    parser.add_argument('--metadisc_alpha', type=float, help='factor for expert meta reward', default=0.01)
    parser.add_argument('--metadisc_dropout_rate', type=float, help='Meta discriminator dropout rate', default=0.3)

    # AC model parameters
    parser.add_argument('--ac_hidden', type=int, nargs='*', help='ActorCritic hidden dim parameters', default=[512, 256])
    parser.add_argument('--actor_dropout_rate', type=float, help='actor dropout rate', default=0.3)
    parser.add_argument('--critic_dropout_rate', type=float, help='critic dropout rate', default=0.3)
    parser.add_argument('--act_dropout', type=float, help='action dropout rate', default=0.3)
    parser.add_argument('--ac_lr', type=float, help='learning rate for actor-critic', default=0.0005)
    parser.add_argument('--ac_update_delay', type=int, help='Starts AC model updates after N demonstration updates', default=20)
    parser.add_argument('--ent_weight', type=float, help='weight factor for entropy loss', default=0.001)
    parser.add_argument('--gamma', type=float, help='reward discount factor', default=0.99)
    parser.add_argument('--target_update', type=float, help='update ratio of target network', default=0.05)

    # other training parameters
    parser.add_argument('--seed', type=int, help='Random seed (default: 1023)', default=1023)
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU to train model", default=False)
    parser.add_argument('--gpu', type=int, help='gpu device (default: 0)', default=0)
    parser.add_argument('--train_batch_size', type=int, help='batch size for training step (default: 256)', default=256)
    parser.add_argument('--eval_batch_size', type=int, help='batch size for evaluation step (default: 5)', default=5)
    parser.add_argument('--epochs', type=int, help='Max number of epochs over the entire training set (default: 100)', default=100)
    parser.add_argument('--every_print', type=int, help='Number of step to print losses', default=30)
    parser.add_argument('--save_every', type=int, help='Save the model every # epochs', default=1)

    # # evaluation parameters
    # parser.add_argument('--run_eval', action="store_true", help='whether to run evaluation with valid dataset', default=False)
    # parser.add_argument('--beam_size', type=int, help='size of beam used in beam search inference (default: 250)', default=250)
    # parser.add_argument('--topk', type=int, help='top ranked diseases to recommend', default=50)

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
    folder_name = 'ADAC_model'
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

    ## start to train ADAC model
    logger.info('Start to train ADAC model')
    train(args)