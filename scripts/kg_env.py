## This script is modified from the code used in the original ADAC RL model (Zhao et al. doi: 10.1145/3397271.3401171)
import torch
import utils
import numpy as np

class KGEnvironment(object):
    def __init__(self, args, pre_train_model, kg, max_path_len=3, state_pre_history=1):
        self.args = args
        self.max_num_nodes = max_path_len + 1
        self.kg = kg
        self.state_pre_history = state_pre_history
        self.pre_train_model = pre_train_model
        # self.reward_shaping_threshold = args.reward_shaping_threshold
        self.pre_train_model_embeddings = utils.get_graphsage_embedding(args)
        self.pre_train_model_embeddings.requires_grad = False
        # self.pre_train_model_embeddings = self.kg.entity_embeddings.weight.clone().detach().type(torch.float)
        # self.pre_train_model_embeddings.requires_grad = False

        self.pre_train_model.model.eval()
        utils.detach_module(self.pre_train_model.model)

        ## Current episode information.
        self._batch_path = None
        self._batch_curr_action_spaces = None
        self._batch_curr_state = None
        self._batch_curr_reward = None
        self._done = False

    def reset(self):
        self._batch_path = None
        self._batch_curr_action_spaces = None
        self._batch_curr_state = None
        self._batch_curr_reward = None
        self._done = False    

    def pad_and_cat_action_space(self, action_spaces, inv_offset):
        db_r_space, db_e_space, db_action_mask = [], [], []
        for (r_space, e_space), action_mask in action_spaces:
            db_r_space.append(r_space)
            db_e_space.append(e_space)
            db_action_mask.append(action_mask)
        r_space = utils.pad_and_cat(db_r_space, padding_value=self.kg.dummy_r)[inv_offset]
        e_space = utils.pad_and_cat(db_e_space, padding_value=self.kg.dummy_e)[inv_offset]
        action_mask = utils.pad_and_cat(db_action_mask, padding_value=0)[inv_offset]
        action_space = ((r_space, e_space), action_mask)
        return action_space

    def validate_action_mask(self, action_mask):
        action_mask_min = action_mask.min()
        action_mask_max = action_mask.max()
        assert (action_mask_min == 0 or action_mask_min == 1)
        assert (action_mask_max == 0 or action_mask_max == 1)
        assert (not any(action_mask.sum(dim=1) == 0))

    def apply_action_masks(self, action_space, last_r, seen_nodes, kg):
        (r_space, e_space), action_mask = action_space

        # forbid self-loop action in 1th-hop
        if seen_nodes.shape[1] == 1:
            action_mask = (1 - (r_space == kg.self_edge).float()) * action_mask
            self.validate_action_mask(action_mask)

        # if agent reachs to self-loop, then stop explore
        if seen_nodes.shape[1] != 1:
            stop_mask = (last_r == kg.self_edge).unsqueeze(1).float()
            action_mask = (1 - stop_mask) * action_mask + stop_mask * (r_space == kg.self_edge).float()
            self.validate_action_mask(action_mask)

        # prevent choosing the seen nodes
        loop_mask = (((seen_nodes.unsqueeze(1) == e_space.unsqueeze(2)).sum(2) > 0) * (r_space != kg.self_edge)).float()
        action_mask *= (1 - loop_mask)
        self.validate_action_mask(action_mask)

        return (r_space, e_space), action_mask

    def _batch_get_actions(self, batch_path):
        """
        Get actions for current node.
        """
        if self._done:
            return None, None

        db_action_spaces, db_references = [], []
        e = batch_path[1][:,-1]
        last_r = batch_path[0][:,-1]
        seen_nodes = batch_path[1]
        entity2bucketid = self.kg.entity2bucketid[e.tolist()]
        key1 = entity2bucketid[:, 0]
        key2 = entity2bucketid[:, 1]
        batch_ref = {}
        for i in range(len(e)):
            key = int(key1[i])
            if not key in batch_ref:
                batch_ref[key] = []
            batch_ref[key].append(i)
        for key in batch_ref:
            action_space = self.kg.action_space_buckets[key]
            l_batch_refs = batch_ref[key]
            g_bucket_ids = key2[l_batch_refs].tolist()
            if self.args.use_gpu:
                r_space_b = action_space[0][0][g_bucket_ids]
            else:
                r_space_b = action_space[0][0][g_bucket_ids].cpu()
            if self.args.use_gpu:
                e_space_b = action_space[0][1][g_bucket_ids]
            else:
                e_space_b = action_space[0][1][g_bucket_ids].cpu()
            if self.args.use_gpu:
                action_mask_b = action_space[1][g_bucket_ids]
            else:
                action_mask_b = action_space[1][g_bucket_ids].cpu()
            if self.args.use_gpu:
                last_r_b = last_r[l_batch_refs].to(self.args.device)
            else:
                last_r_b = last_r[l_batch_refs]
            if self.args.use_gpu:
                seen_nodes_b = seen_nodes[l_batch_refs].to(self.args.device)
            else:
                seen_nodes_b = seen_nodes[l_batch_refs]
            action_space_b = ((r_space_b, e_space_b), action_mask_b)
            action_space_b = self.apply_action_masks(action_space_b, last_r_b, seen_nodes_b, self.kg)
            db_action_spaces.append(action_space_b)
            db_references.extend(l_batch_refs)
        inv_offset = [i for i, _ in sorted(enumerate(db_references), key=lambda x: x[1])]
        action_space = self.pad_and_cat_action_space(db_action_spaces, inv_offset)

        return action_space

    def _batch_get_state(self, batch_path):
        path_len = batch_path[0].shape[1]
        return [batch_path[1][:,0], batch_path[0][:,max(0, path_len-self.state_pre_history-1):], batch_path[1][:,max(0, path_len-self.state_pre_history-1):]]

    def prob(self, e1, pred_e2):
        if self.args.use_gpu is True:
            X = torch.cat([self.pre_train_model_embeddings[e1],self.pre_train_model_embeddings[pred_e2]], dim=1).to(self.args.device)
        else:
            X = torch.cat([self.pre_train_model_embeddings[e1],self.pre_train_model_embeddings[pred_e2]], dim=1)
        # return torch.tensor(self.pre_train_model.predict_proba(X)[:, 1])
        return torch.tensor((np.argmax(self.pre_train_model.predict_proba(X), axis=1) == 1).astype(float) * self.pre_train_model.predict_proba(X)[:,1])

    def reward_fun(self, e1, e2, pred_e2):
        if self.args.use_gpu:
            hit_disease_reward = ((pred_e2.to(self.args.device).unsqueeze(1) == torch.tensor(self.args.disease_ids).to(self.args.device)).sum(dim=1) > 0).float()
            real_reward = self.prob(e1, pred_e2).to(self.args.device)
            # real_reward_mask = (real_reward >= self.reward_shaping_threshold).float().to(self.args.device)
            # real_reward = real_reward * real_reward_mask * hit_disease_reward
            real_reward[hit_disease_reward == 0] = -1.0 ## if the last node is not disease, set -1.0
            binary_reward = (pred_e2.to(self.args.device) == e2.to(self.args.device)).float()
            return binary_reward * self.args.tp_reward + (1 - binary_reward) * real_reward
        else:
            hit_disease_reward = ((pred_e2.unsqueeze(1) == torch.tensor(self.args.disease_ids)).sum(dim=1) > 0).float()
            real_reward = self.prob(e1, pred_e2)
            # real_reward_mask = (real_reward >= self.reward_shaping_threshold).float()
            # real_reward = real_reward * real_reward_mask * hit_disease_reward
            real_reward[hit_disease_reward == 0] = -1.0 ## if the last node is not disease, set -1.0
            binary_reward = (pred_e2 == e2).float()
            return binary_reward * self.args.tp_reward + (1 - binary_reward) * real_reward

    def _batch_get_reward(self, batch_path):

        if not self._done:
            return np.zeros(batch_path[1].shape[0])

        if self.target_ids is not None:
            source_ids = batch_path[1][:,0]        
            target_ids = self.target_ids
            pred_ids = batch_path[1][:,-1]
            batch_reward = self.reward_fun(source_ids, target_ids, pred_ids)

            if self.args.use_gpu:
                return batch_reward.cpu().numpy()
            else:
                return batch_reward.numpy()
        else:
            print(f"Warning: No target ids provided to calculate reward")
            return np.zeros(batch_path[1].shape[0])

    def _is_done(self):
        """Episode ends only if max path length is reached."""
        return self._done or self._batch_path[0].shape[1] >= self.max_num_nodes

    def initialize_path(self, source_ids, target_ids=None):

        if type(source_ids) is not torch.Tensor:
            source_ids = torch.tensor(source_ids)
        if target_ids is not None:
            if type(target_ids) is not torch.Tensor:
                self.target_ids = torch.tensor(target_ids)
            else:
                self.target_ids = target_ids
        else:
            self.target_ids = target_ids

        self._batch_path = [torch.tensor([self.kg.self_edge] * len(source_ids)).view(-1,1), source_ids.view(-1,1)]
        self._done = False
        self._batch_curr_state = self._batch_get_state(self._batch_path)
        if self.args.use_gpu:
            utils.empty_gpu_cache(self.args)
        self._batch_curr_action_spaces = self._batch_get_actions(self._batch_path)
        if self.args.use_gpu:
            utils.empty_gpu_cache(self.args)
        self._batch_curr_reward = self._batch_get_reward(self._batch_path)
        if self.args.use_gpu:
            utils.empty_gpu_cache(self.args)

    def batch_step(self, true_next_act, offset=None):
        next_relation_ids, next_entity_ids = true_next_act

        def offset_path_history(p, offset):
            for i, x in enumerate(p):
                p[i] = x[offset, :]

        if offset is not None:
            offset_path_history(self._batch_path, offset)

        assert len(next_relation_ids) == len(self._batch_path[0])
        assert len(next_entity_ids) == len(self._batch_path[0])

        self._batch_path[0] = torch.cat([self._batch_path[0],next_relation_ids.cpu().view(-1,1)], dim=1)
        self._batch_path[1] = torch.cat([self._batch_path[1],next_entity_ids.cpu().view(-1,1)], dim=1)
        self._done = self._is_done()
        self._batch_curr_state = self._batch_get_state(self._batch_path)
        if self.args.use_gpu:
            utils.empty_gpu_cache(self.args)
        self._batch_curr_action_spaces = self._batch_get_actions(self._batch_path)
        if self.args.use_gpu:
            utils.empty_gpu_cache(self.args)
        self._batch_curr_reward = self._batch_get_reward(self._batch_path)
        if self.args.use_gpu:
            utils.empty_gpu_cache(self.args)
