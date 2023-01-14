## This script is modified from the code used in the MultiHop model (Lin et al. doi: 10.48550/arXiv.1808.10568) and accessed from Github (https://github.com/salesforce/MultiHopKG)
import os
import random
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import utils
import copy
import pickle
from tqdm import trange
from torch.distributions import Categorical


class LFramework(nn.Module):
    def __init__(self, args, kg, pn):
        super(LFramework, self).__init__()
        self.args = args
        self.data_dir = args.data_dir
        self.model_path = args.model_path
        self.use_gpu = args.use_gpu

        # Training hyperparameters
        self.train_batch_size = args.train_batch_size
        self.val_batch_size = args.eval_batch_size
        self.num_epochs = args.num_epochs
        self.num_wait_epochs = args.num_wait_epochs
        self.num_peek_epochs = args.num_peek_epochs
        self.num_check_epochs = args.num_check_epochs
        self.learning_rate = args.lr
        self.grad_norm = args.grad_norm
        self.optim = None

        self.kg = kg
        self.pn = pn

    def print_all_model_parameters(self):
        print('\nModel Parameters', flush=True)
        print('--------------------------', flush=True)
        for name, param in self.named_parameters():
            print(name, param.numel(), 'requires_grad={}'.format(param.requires_grad))
        param_sizes = [param.numel() for param in self.parameters()]
        print('Total # parameters = {}'.format(sum(param_sizes)), flush=True)
        print('--------------------------', flush=True)

    def run_train(self, train_data, eval_drug_disease_dict, all_drug_disease_dict):
        self.print_all_model_parameters()

        if self.optim is None:
            self.optim = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)

        writer = SummaryWriter(log_dir=os.path.join(self.model_path, 'tensorboard_runs'))
        # Track val metrics changes
        best_eval_metrics = 0
        best_avg_pred_score = 0
        best_epoch_id = 0
        eval_metrics_history = []
        best_model_evaluation_result = None
        best_model_state_dict = None
        train_dataloader = utils.ACDataLoader(list(range(train_data.shape[0])), self.train_batch_size) 

        for epoch in range(1, self.num_epochs + 1):

            self.args.logger.info(f'running epoch{epoch}...')

            # Update model parameters
            self.train()
            self.fn.model.eval()

            train_dataloader.reset()
            batch_losses = []
            entropies = []
            rewards = []
            ave_success_hit = []

            pbar = tqdm(total=train_dataloader.num_paths)
            while train_dataloader.has_next():

                self.optim.zero_grad()
                batch_path_id = train_dataloader.get_batch()
                mini_batch = train_data[batch_path_id]
                if len(mini_batch) < self.train_batch_size:
                    continue
                loss = self.loss(mini_batch)
                loss['model_loss'].backward()
                if self.grad_norm > 0:
                    clip_grad_norm_(self.parameters(), self.grad_norm)

                self.optim.step()

                batch_losses.append(loss['print_loss'])
                if 'entropy' in loss:
                    entropies.append(loss['entropy'])
                if 'reward' in loss:
                    rewards.append(loss['reward'])
                if 'ave_success_hit' in loss:
                    ave_success_hit.append(loss['ave_success_hit'])
                del mini_batch
                utils.empty_gpu_cache(self.args)
                pbar.update(len(batch_path_id))

            # Check training statistics
            mean_batch_loss = np.mean(batch_losses)
            stdout_msg = 'Epoch {}: average training loss = {:.5f}'.format(epoch, mean_batch_loss)
            if entropies:
                mean_entropy = np.mean(entropies)
                stdout_msg += ' entropy = {:.5f}'.format(mean_entropy)
            if rewards:
                mean_rewards = np.mean(rewards)
                stdout_msg += ' reward = {:.5f}'.format(mean_rewards)
            if ave_success_hit:
                mean_ave_success_hits = np.mean(ave_success_hit)
                stdout_msg += ' average sucessful hits = {:.5f}'.format(mean_ave_success_hits)
            self.args.logger.info(stdout_msg)  

            # Check val set performance
            if epoch % self.num_peek_epochs == 0:
                results = self.evaluate(eval_drug_disease_dict, all_drug_disease_dict, save_paths=True)
                if self.args.use_gpu:
                    utils.empty_gpu_cache(self.args)
                avg_pred_score = results['avg_pred_score']
                avg_hit = results['avg_hit']
                writer.add_scalars('Average Prediction Score for Evaluation Data', {'avg_pred_score': avg_pred_score}, epoch)
                writer.add_scalars('Average Hit Ratio for Evaluation Data', {'avg_hit': avg_hit}, epoch)

                # Save checkpoint
                if avg_hit >= best_eval_metrics:
                    if avg_hit == best_eval_metrics:
                        if avg_pred_score > best_avg_pred_score:
                            best_model_state_dict = copy.deepcopy(self.state_dict())
                            best_eval_metrics = avg_hit
                            best_avg_pred_score = avg_pred_score
                            best_epoch_id = epoch
                            best_model_evaluation_result = results
                    else:
                        best_model_state_dict = copy.deepcopy(self.state_dict())
                        best_eval_metrics = avg_hit
                        best_avg_pred_score = avg_pred_score
                        best_epoch_id = epoch
                        best_model_evaluation_result = results
                else:
                    # Early stopping
                    if epoch >= self.num_wait_epochs and avg_hit < np.mean(eval_metrics_history[-self.num_check_epochs:]):
                        break
                eval_metrics_history.append(avg_hit)

        ## Saves model and its weights
        with open(os.path.join(self.args.pred_paths_save_location,'eval_pred_paths.pkl'),'wb') as outfile:
            pickle.dump(best_model_evaluation_result, outfile)
        self.save_checkpoint(epoch_id=best_epoch_id, state_dict=best_model_state_dict)

    def format_batch(self, batch_data, num_labels=-1, num_tiles=1):
        """
        Convert batched tuples to the tensors accepted by the NN.
        """
        def convert_to_binary_multi_source(source):
            if self.use_gpu:
                source_label = utils.zeros_var_cuda([len(source), num_labels], args=self.args, use_gpu=True)
            else:
                source_label = utils.zeros_var_cuda([len(source), num_labels], args=self.args, use_gpu=False)
            for i in range(len(source)):
                source_label[i][source[i]] = 1
            return source_label

        def convert_to_binary_multi_target(target):
            if self.use_gpu:
                target_label = utils.zeros_var_cuda([len(target), num_labels], args=self.args, use_gpu=True)
            else:
                target_label = utils.zeros_var_cuda([len(target), num_labels], args=self.args, use_gpu=False)
            for i in range(len(target)):
                target_label[i][target[i]] = 1
            return target_label

        batch_source, batch_target = [], []
        for i in range(len(batch_data)):
            source, target = batch_data[i]
            batch_source.append(source)
            batch_target.append(target)
        if self.use_gpu:
            batch_source = utils.var_cuda(torch.LongTensor(batch_source), requires_grad=False, args=self.args, use_gpu=True)
        else:
            batch_source = utils.var_cuda(torch.LongTensor(batch_source), requires_grad=False, args=self.args, use_gpu=False)
        if type(batch_target[0]) is list:
            batch_target = convert_to_binary_multi_target(batch_target)
        elif type(batch_source[0]) is list:
            batch_source = convert_to_binary_multi_source(batch_source)
        else:
            if self.use_gpu: 
                batch_target = utils.var_cuda(torch.LongTensor(batch_target), requires_grad=False, args=self.args, use_gpu=True)
            else:
                batch_target = utils.var_cuda(torch.LongTensor(batch_target), requires_grad=False, args=self.args, use_gpu=False)
        # Rollout multiple times for each example
        if num_tiles > 1:
            batch_source = utils.tile_along_beam(batch_source, num_tiles)
            batch_target = utils.tile_along_beam(batch_target, num_tiles)
        return batch_source, batch_target

    def make_full_batch(self, mini_batch, batch_size, multi_answers=False):
        dummy_e = self.kg.dummy_e
        dummy_r = self.kg.dummy_r
        if multi_answers:
            dummy_example = (dummy_e, [dummy_e], dummy_r)
        else:
            dummy_example = (dummy_e, dummy_e, dummy_r)
        for _ in range(batch_size - len(mini_batch)):
            mini_batch.append(dummy_example)

    def save_checkpoint(self, epoch_id, state_dict):
        """
        Save model checkpoint.
        """
        checkpoint_dict = dict()
        checkpoint_dict['state_dict'] = state_dict
        checkpoint_dict['epoch_id'] = epoch_id

        self.best_path = os.path.join(self.model_path, f'_checkpoint{epoch_id}_model_best.tar')
        torch.save(checkpoint_dict, self.best_path)
        print(f"The best model is saved to \'{self.best_path}\'", flush=True)

    def load_checkpoint(self, input_file):
        """
        Load model checkpoint.
        """
        if os.path.isfile(input_file):
            print(f"Loading model from \'{input_file}\'", flush=True)
            if self.use_gpu:
                checkpoint = torch.load(input_file, map_location="cuda:{}".format(self.args.gpu))
            else:
                checkpoint = torch.load(input_file)
            self.load_state_dict(checkpoint['state_dict'])
        else:
            print(f"No checkpoint found at \'{input_file}\'")


class PolicyGradient(LFramework):
    def __init__(self, args, kg, pn):
        super(PolicyGradient, self).__init__(args, kg, pn)

        # Training hyperparameters
        # self.use_action_space_bucketing = args.use_action_space_bucketing
        self.num_rollouts = args.num_rollouts
        self.num_rollout_steps = args.num_rollout_steps
        self.beta = args.beta  # entropy regularization parameter
        self.gamma = args.gamma  # shrinking factor
        self.action_dropout_rate = args.action_dropout_rate

        # Inference hyperparameters
        self.beam_size = args.beam_size

        # Analysis
        self.path_types = dict()
        self.num_path_types = 0

    def reward_fun(self, e1, e2, pred_e2):
        return (pred_e2 == e2).float()

    def loss(self, mini_batch):
        
        # def stablize_reward(r):
        #     r_2D = r.view(-1, self.num_rollouts)
        #     # if self.baseline == 'avg_reward':
        #     #     stabled_r_2D = r_2D - r_2D.mean(dim=1, keepdim=True)
        #     # elif self.baseline == 'avg_reward_normalized':
        #     stabled_r_2D = (r_2D - r_2D.mean(dim=1, keepdim=True)) / (r_2D.std(dim=1, keepdim=True) + utils.EPSILON)
        #     # else:
        #     #     raise ValueError('Unrecognized baseline function: {}'.format(self.baseline))
        #     stabled_r = stabled_r_2D.view(-1)
        #     return stabled_r
    
        e1, e2 = self.format_batch(mini_batch, num_tiles=self.num_rollouts)
        output = self.rollout(e1, num_steps=self.num_rollout_steps)
        utils.empty_gpu_cache(self.args)

        # Compute policy gradient loss
        pred_e2 = output['pred_e2']
        log_action_probs = output['log_action_probs']
        action_entropy = output['action_entropy']
        # ## find the indexes of last non-self-loop relation in the path
        # final_reward_index = torch.sum(1- (torch.cat([x.unsqueeze(1) for x in output['path_relation']], dim=1) == self.kg.self_edge).int(), dim=1)-1
        # final_reward_index[final_reward_index==-1] = 0
        # final_reward_index = final_reward_index.tolist()


        # Compute discounted reward
        final_reward = self.reward_fun(e1, e2, pred_e2)
        utils.empty_gpu_cache(self.args)
        if self.args.use_gpu:
            cum_discounted_rewards = torch.zeros(len(final_reward), self.num_rollout_steps).to(self.args.device)
        else:
            pass
        # rows_index = range(0, len(final_reward))
        # cols_index = final_reward_index
        # cum_discounted_rewards[rows_index,cols_index] = final_reward
        cum_discounted_rewards[:,-1] = final_reward

        R = 0
        for i in range(self.num_rollout_steps - 1, -1, -1):
            R = self.gamma * R + cum_discounted_rewards[:,i]
            cum_discounted_rewards[:,i] = R

        # Compute all loss functions
        actor_loss, critic_loss = 0, 0
        # values = torch.cat(self.mdl.values,dim=1)

        for i in range(self.num_rollout_steps):
            log_action_prob = log_action_probs[i]
            # advantage = cum_discounted_rewards[:,i] - values[:,i]
            # actor_loss += - log_action_prob * advantage.detach()
            # critic_loss += advantage.pow(2)
            actor_loss += - log_action_prob * cum_discounted_rewards[:,i]

        # Entropy regularization
        entropy = torch.cat([x.unsqueeze(1) for x in action_entropy], dim=1).mean(dim=1)
        actor_loss = (actor_loss - entropy * self.beta).mean()
        # critic_loss = critic_loss/self.num_rollout_steps
        # critic_loss = critic_loss.mean()
        # loss = actor_loss + critic_loss
        loss = actor_loss

        loss_dict = {}
        loss_dict['model_loss'] = loss
        loss_dict['print_loss'] = float(loss)
        loss_dict['reward'] = float(final_reward.mean())
        loss_dict['entropy'] = float(entropy.mean())
        loss_dict['ave_success_hit'] = float((pred_e2 == e2).float().mean())

        return loss_dict

    def rollout(self, e_s, num_steps):

        assert (num_steps > 0)
        kg, pn = self.kg, self.pn

        # Initialization
        log_action_probs = []
        action_entropy = []
        path_r = []
        if self.use_gpu:
            r_s = utils.int_fill_var_cuda(e_s.size(), kg.self_edge, args=self.args, use_gpu=True)
        else:
            r_s = utils.int_fill_var_cuda(e_s.size(), kg.self_edge, args=self.args, use_gpu=False)

        seen_nodes = e_s.unsqueeze(1)
        pn.initialize_path((r_s, e_s), kg)
        path_trace = [(r_s, e_s)]

        for t in range(num_steps):
            last_r, e = path_trace[-1]
            obs = [e_s, t==(num_steps-1), last_r, seen_nodes]
            db_outcomes, inv_offset, policy_entropy = pn.transit(e, obs, kg)
            sample_outcome = self.sample_action(db_outcomes, inv_offset)
            action = sample_outcome['action_sample']
            pn.update_path(action, kg)
            action_prob = sample_outcome['action_prob']
            log_action_probs.append(utils.safe_log(action_prob))
            action_entropy.append(policy_entropy)
            seen_nodes = torch.cat([seen_nodes, e.unsqueeze(1)], dim=1)
            path_trace.append(action)
            path_r.append(action[0])


        pred_e2 = path_trace[-1][1]

        return {
            'pred_e2': pred_e2,
            'log_action_probs': log_action_probs,
            'action_entropy': action_entropy,
            'path_relation': path_r
        }

    def sample_action(self, db_outcomes, inv_offset=None):
        """
        Sample an action based on current policy.
        :param db_outcomes (((r_space, e_space), action_mask), action_dist):
                r_space: (Variable:batch) relation space
                e_space: (Variable:batch) target entity space
                action_mask: (Variable:batch) binary mask indicating padding actions.
                action_dist: (Variable:batch) action distribution of the current step based on set_policy
                    network parameters
        :param inv_offset: Indexes for restoring original order in a batch.
        :return next_action (next_r, next_e): Sampled next action.
        :return action_prob: Probability of the sampled action.
        """

        # def apply_action_dropout_mask(action_dist, action_mask):
        #     if self.action_dropout_rate > 0:
        #         dropout_action_mask = copy.deepcopy(action_mask)
        #         for index, row in enumerate(dropout_action_mask):
        #             print(index, flush=True)
        #             one_index = torch.where(row==1)[0]
        #             one_index = one_index[torch.randperm(len(one_index))]
        #             num_zeros = int(len(one_index)*self.action_dropout_rate)
        #             row[one_index[:num_zeros]] = 0
        #         sample_action_dist = action_dist * dropout_action_mask
        #         # rand = torch.rand(action_dist.size())
        #         # if self.use_gpu:
        #         #     action_keep_mask = utils.var_cuda(rand > self.action_dropout_rate, use_gpu=True).float()
        #         # else:
        #         #     action_keep_mask = utils.var_cuda(rand > self.action_dropout_rate, use_gpu=False).float()
        #         # # There is a small chance that that action_keep_mask is accidentally set to zero.
        #         # # When this happen, we take a random sample from the available actions.
        #         # # sample_action_dist = action_dist * (action_keep_mask + ops.EPSILON)
        #         # sample_action_dist = action_dist * action_keep_mask + utils.EPSILON * (1 - action_keep_mask) * action_mask
        #         return sample_action_dist
        #     else:
        #         return action_dist

        def apply_action_dropout_probs(action_dist):
            if self.action_dropout_rate > 0:
                copy_probs = copy.deepcopy(action_dist.detach())
                for row in copy_probs:
                    one_index = torch.where(row!=0)[0]
                    one_index = one_index[torch.randperm(len(one_index))]
                    num_zeros = int(len(one_index)*self.action_dropout_rate)
                    row[one_index[:num_zeros]] = 0
                return copy_probs
            else:
                return action_dist

        def sample(action_space, action_dist):
            sample_outcome = {}
            ((r_space, e_space), action_mask) = action_space
            # sample_action_dist = apply_action_dropout_mask(action_dist, action_mask)
            sample_action_dist = apply_action_dropout_probs(action_dist)
            m = Categorical(sample_action_dist)
            idx = m.sample().view(-1,1)
            # idx = torch.multinomial(sample_action_dist, 1, replacement=True)
            next_r = utils.batch_lookup(r_space, idx)
            next_e = utils.batch_lookup(e_space, idx)
            action_prob = utils.batch_lookup(action_dist, idx)
            sample_outcome['action_sample'] = (next_r, next_e)
            sample_outcome['action_prob'] = action_prob
            return sample_outcome

        if inv_offset is not None:
            next_r_list = []
            next_e_list = []
            action_dist_list = []
            action_prob_list = []
            for action_space, action_dist in db_outcomes:
                sample_outcome = sample(action_space, action_dist)
                next_r_list.append(sample_outcome['action_sample'][0])
                next_e_list.append(sample_outcome['action_sample'][1])
                action_prob_list.append(sample_outcome['action_prob'])
                action_dist_list.append(action_dist)
            next_r = torch.cat(next_r_list, dim=0)[inv_offset]
            next_e = torch.cat(next_e_list, dim=0)[inv_offset]
            action_sample = (next_r, next_e)
            action_prob = torch.cat(action_prob_list, dim=0)[inv_offset]
            sample_outcome = {}
            sample_outcome['action_sample'] = action_sample
            sample_outcome['action_prob'] = action_prob
        else:
            sample_outcome = sample(db_outcomes[0][0], db_outcomes[0][1])

        return sample_outcome


    def evaluate(self, drug_disease_dict, all_drug_disease_dict, save_paths=False):

        self.eval()
        eval_drugs = torch.tensor(list(drug_disease_dict.keys()))
        self.args.logger.info('Evaluating model')
        with torch.no_grad():
            ## predict paths for drugs
            eval_dataloader = utils.ACDataLoader(list(range(len(eval_drugs))), self.args.eval_batch_size)
            pbar = tqdm(total=eval_dataloader.num_paths)
            if save_paths:
                all_paths_r, all_paths_e, all_prob_scores = [], [], []
            pred_diseases = dict()
            while eval_dataloader.has_next():
                dids_idx = eval_dataloader.get_batch()
                source_ids = eval_drugs[dids_idx].to(self.args.device)
                res = utils.beam_search(self.args, self.pn, source_ids, self.kg, self.num_rollout_steps, self.beam_size)
                assert (res['output_beam_size'] == self.beam_size)
                if self.args.use_gpu:
                    utils.empty_gpu_cache(self.args)
                if save_paths:
                    all_paths_r += [res['paths'][0]]
                    all_paths_e += [res['paths'][1]]
                    all_prob_scores += [res['pred_prob_scores']]
                for index in range(0,res['paths'][1].shape[0], self.args.beam_size):
                    drug_id = int(res['paths'][1][index][0])
                    pred_list = res['paths'][1][index:(index+self.args.beam_size),-1].tolist()
                    disease_list = list(set(pred_list).intersection(set(self.args.disease_ids)))
                    pred_diseases[drug_id] = dict()
                    pred_diseases[drug_id]['list'] = disease_list
                    if len(disease_list) !=0:
                        pred_diseases[drug_id]['pred_score'] = self.prob([drug_id]*len(disease_list),disease_list)
                    else:
                        pred_diseases[drug_id]['pred_score'] = torch.tensor([])
                pbar.update(len(source_ids))
        
            failed_pred = 0
            avg_pred_score = []
            recalls, precisions, hits = [], [], []
            all_recalls, all_precisions, all_hits = [], [], []
            avg_disease_num = []
            hit_disease_target_pairs = dict()
            for drug_id in drug_disease_dict.keys():
                if len(pred_diseases[drug_id]['list']) == 0:
                    failed_pred += 1
                    continue
                avg_disease_num += [len(pred_diseases[drug_id]['list'])]
                pred_list, rel_set, all_rel_set = pred_diseases[drug_id]['list'], drug_disease_dict[drug_id], all_drug_disease_dict[drug_id]

                hit_num = len(set(pred_list).intersection(set(rel_set)))
                if hit_num > 0:
                    hit_disease_target_pairs[drug_id] = list(set(pred_list).intersection(set(rel_set)))
                recall = hit_num / len(rel_set)
                precision = hit_num / len(pred_list)
                hit = 1.0 if hit_num > 0 else 0.0
                recalls.append(recall)
                precisions.append(precision)
                hits.append(hit)

                all_hit_num = len(set(pred_list).intersection(set(all_rel_set)))
                all_recall = all_hit_num / len(all_rel_set)
                all_precision = all_hit_num / len(pred_list)
                all_hit = 1.0 if all_hit_num > 0 else 0.0
                all_recalls.append(all_recall)
                all_precisions.append(all_precision)
                all_hits.append(all_hit)
                avg_pred_score.append(pred_diseases[drug_id]['pred_score'].mean().item())

            self.args.logger.info(f'{failed_pred}/{len(drug_disease_dict.keys())} from evaluation dataset have no disease prediction')
            self.args.logger.info(f'Average hitting #diseases {sum(avg_disease_num)/len(avg_disease_num)}')
            avg_pred_score = np.mean(avg_pred_score)
            avg_recall = np.mean(recalls) * 100
            avg_precision = np.mean(precisions) * 100
            avg_hit = np.mean(hits) * 100
            self.args.logger.info(f'Avg prediction score={avg_pred_score:.3f}')
            self.args.logger.info(f'Evaluation dataset only: Recall={avg_recall:.3f} | HR={avg_hit:.3f} | Precision={avg_precision:.3f}')
            all_avg_recall = np.mean(all_recalls) * 100
            all_avg_precision = np.mean(all_precisions) * 100
            all_avg_hit = np.mean(all_hits) * 100
            self.args.logger.info(f'all datasets (train, val and test): Recall={all_avg_recall:.3f} | HR={all_avg_hit:.3f} | Precision={all_avg_precision:.3f}')

            if save_paths:
                all_paths_r = torch.cat(all_paths_r)
                all_paths_e = torch.cat(all_paths_e)
                all_prob_scores = torch.cat(all_prob_scores)
                return {'paths': [all_paths_r.cpu(),all_paths_e.cpu()], 'prob_scores': all_prob_scores, 'hit_pairs': hit_disease_target_pairs, "avg_hit": avg_hit, "avg_pred_score": avg_pred_score}
            else:
                return None
