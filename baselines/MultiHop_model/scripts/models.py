## This script is modified from the code used in the MultiHop model (Lin et al. doi: 10.48550/arXiv.1808.10568) and accessed from Github (https://github.com/salesforce/MultiHopKG)
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from model_framework import PolicyGradient

class GraphSearchPolicy(nn.Module):
    def __init__(self, args):
        super(GraphSearchPolicy, self).__init__()
        # self.model = args.model

        self.args = args
        self.history_dim = args.history_dim
        self.history_num_layers = args.history_num_layers
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        self.action_dim = args.entity_dim + args.relation_dim
        self.ff_dropout_rate = args.ff_dropout_rate
        self.action_dropout_rate = args.action_dropout_rate

        self.xavier_initialization = args.xavier_initialization

        self.path = None

        # Set policy network modules
        self.define_modules()
        self.initialize_modules()

    def transit(self, e, obs, kg, merge_aspace_batching_outcome=False):
        """
        Compute the next action distribution based on
            (a) the current node (entity) in KG and the query relation
            (b) action history representation
        """
        e_s, last_step, last_r, seen_nodes = obs

        # Representation of the current state (current node and other observations)
        # Q = kg.get_relation_embeddings(q)
        H = self.path[-1][0][-1, :, :]
        E = kg.get_entity_embeddings(e)
        # X = torch.cat([E, H, Q], dim=-1)
        X = torch.cat([E, H], dim=-1)

        # MLP for hidden state
        X = self.W1(X)
        X = F.relu(X)
        X = self.W1Dropout(X)
        X = self.W2(X)
        X = F.relu(X)
        X2 = self.W2Dropout(X)

        def policy_nn_fun(X2, action_space):
            (r_space, e_space), action_mask = action_space
            A = self.get_action_embedding((r_space, e_space), kg)
            action_dist = F.softmax(torch.squeeze(A @ torch.unsqueeze(X2, 2), 2) - (1 - action_mask) * utils.HUGE_INT, dim=-1)

            return action_dist, utils.entropy(action_dist)

        def pad_and_cat_action_space(action_spaces, inv_offset):
            db_r_space, db_e_space, db_action_mask = [], [], []
            for (r_space, e_space), action_mask in action_spaces:
                db_r_space.append(r_space)
                db_e_space.append(e_space)
                db_action_mask.append(action_mask)
            r_space = utils.pad_and_cat(db_r_space, padding_value=kg.dummy_r)[inv_offset]
            e_space = utils.pad_and_cat(db_e_space, padding_value=kg.dummy_e)[inv_offset]
            action_mask = utils.pad_and_cat(db_action_mask, padding_value=0)[inv_offset]
            action_space = ((r_space, e_space), action_mask)
            return action_space

        db_outcomes = []
        entropy_list = []
        references = []
        db_action_spaces, db_references = self.get_action_space_in_buckets(e, obs, kg)
        for action_space_b, reference_b in zip(db_action_spaces, db_references):
            X2_b = X2[reference_b, :]
            action_dist_b, entropy_b = policy_nn_fun(X2_b, action_space_b) # At X W2ReLU(W1[et;ht;rq])
            references.extend(reference_b)
            db_outcomes.append((action_space_b, action_dist_b))
            entropy_list.append(entropy_b)
        inv_offset = [i for i, _ in sorted(enumerate(references), key=lambda x: x[1])]
        entropy = torch.cat(entropy_list, dim=0)[inv_offset]
        if merge_aspace_batching_outcome:
            db_action_dist = []
            for _, action_dist in db_outcomes:
                db_action_dist.append(action_dist)
            action_space = pad_and_cat_action_space(db_action_spaces, inv_offset)
            action_dist = utils.pad_and_cat(db_action_dist, padding_value=0)[inv_offset]
            db_outcomes = [(action_space, action_dist)]
            inv_offset = None

        return db_outcomes, inv_offset, entropy

    def initialize_path(self, init_action, kg):
        # [batch_size, action_dim]
        init_action_embedding = self.get_action_embedding(init_action, kg)
        init_action_embedding.unsqueeze_(1)
        # [num_layers, batch_size, dim]
        if self.args.use_gpu:
            init_h = utils.zeros_var_cuda([self.history_num_layers, len(init_action_embedding), self.history_dim], args=self.args, use_gpu=True)
        else:
            init_h = utils.zeros_var_cuda([self.history_num_layers, len(init_action_embedding), self.history_dim], args=self.args, use_gpu=False)
        if self.args.use_gpu:
            init_c = utils.zeros_var_cuda([self.history_num_layers, len(init_action_embedding), self.history_dim], args=self.args, use_gpu=True)
        else:
            init_c = utils.zeros_var_cuda([self.history_num_layers, len(init_action_embedding), self.history_dim], args=self.args, use_gpu=False)
        self.path = [self.path_encoder(init_action_embedding, (init_h, init_c))[1]]
        # self.values = []

    def update_path(self, action, kg, offset=None):
        """
        Once an action was selected, update the action history.
        :param action (r, e): (Variable:batch) indices of the most recent action
            - r is the most recently traversed edge;
            - e is the destination entity.
        :param offset: (Variable:batch) if None, adjust path history with the given offset, used for search
        :param KG: Knowledge graph environment.
        """
        def offset_path_history(p, offset):
            for i, x in enumerate(p):
                if type(x) is tuple:
                    new_tuple = tuple([_x[:, offset, :] for _x in x])
                    p[i] = new_tuple
                else:
                    p[i] = x[offset, :]

        # update action history
        action_embedding = self.get_action_embedding(action, kg)
        if offset is not None:
            offset_path_history(self.path, offset)

        self.path.append(self.path_encoder(action_embedding.unsqueeze(1), self.path[-1])[1])

    def get_action_space_in_buckets(self, e, obs, kg, collapse_entities=False):
        """
        To compute the search operation in batch, we group the action spaces of different states
        (i.e. the set of outgoing edges of different nodes) into buckets based on their sizes to
        save the memory consumption of paddings.

        For example, in large knowledge graphs, certain nodes may have thousands of outgoing
        edges while a long tail of nodes only have a small amount of outgoing edges. If a batch
        contains a node with 1000 outgoing edges while the rest of the nodes have a maximum of
        5 outgoing edges, we need to pad the action spaces of all nodes to 1000, which consumes
        lots of memory.

        With the bucketing approach, each bucket is padded separately. In this case the node
        with 1000 outgoing edges will be in its own bucket and the rest of the nodes will suffer
        little from padding the action space to 5.

        Once we grouped the action spaces in buckets, the policy network computation is carried
        out for every bucket iteratively. Once all the computation is done, we concatenate the
        results of all buckets and restore their original order in the batch. The computation
        outside the policy network module is thus unaffected.

        """
        e_s, last_step, last_r, seen_nodes = obs
        assert(len(e) == len(last_r))
        assert(len(e) == len(e_s))
        db_action_spaces, db_references = [], []

        if collapse_entities:
            raise NotImplementedError
        else:
            entity2bucketid = kg.entity2bucketid[e.tolist()]
            key1 = entity2bucketid[:, 0]
            key2 = entity2bucketid[:, 1]
            batch_ref = {}
            for i in range(len(e)):
                key = int(key1[i])
                if not key in batch_ref:
                    batch_ref[key] = []
                batch_ref[key].append(i)
            for key in batch_ref:
                action_space = kg.action_space_buckets[key]
                # l_batch_refs: ids of the examples in the current batch of examples
                # g_bucket_ids: ids of the examples in the corresponding KG action space bucket
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
                e_b = e[l_batch_refs]
                last_r_b = last_r[l_batch_refs]
                e_s_b = e_s[l_batch_refs]
                seen_nodes_b = seen_nodes[l_batch_refs]
                obs_b = [e_s_b, last_step, last_r_b, seen_nodes_b]
                action_space_b = ((r_space_b, e_space_b), action_mask_b)
                action_space_b = self.apply_action_masks(action_space_b, e_b, obs_b, kg, key, g_bucket_ids)
                db_action_spaces.append(action_space_b)
                db_references.append(l_batch_refs)

        return db_action_spaces, db_references

    def get_action_space(self, e, obs, kg):
        r_space, e_space = kg.action_space[0][0][e], kg.action_space[0][1][e]
        action_mask = kg.action_space[1][e]
        action_space = ((r_space, e_space), action_mask)
        return self.apply_action_masks(action_space, e, obs, kg)

    def apply_action_masks(self, action_space, e, obs, kg, key=None, g_bucket_ids=None):
        (r_space, e_space), action_mask = action_space
        e_s, last_step, last_r, seen_nodes = obs

        # # Prevent the agent from selecting the ground truth edge
        # Since the graph edge doesn't have the ground truth edge
        # ground_truth_edge_mask = self.get_ground_truth_edge_mask(e, r_space, e_space, e_s, q, e_t, kg)
        # action_mask -= ground_truth_edge_mask
        # self.validate_action_mask(action_mask)

        # forbid self-loop action in 1th-hop
        if seen_nodes.shape[1] == 1:
            action_mask = (1 - (r_space == kg.self_edge).float()) * action_mask
            self.validate_action_mask(action_mask)

        # if agent reachs to self-loop, then stop explore
        if seen_nodes.shape[1] != 1:
            stop_mask = (last_r == kg.self_edge).unsqueeze(1).float()
            action_mask = (1 - stop_mask) * action_mask + stop_mask * (r_space == kg.self_edge).float()
            self.validate_action_mask(action_mask)

        # prevent loops
        loop_mask = (((seen_nodes.unsqueeze(1) == e_space.unsqueeze(2)).sum(2) > 0) * (r_space != kg.self_edge)).float()
        action_mask *= (1 - loop_mask)
        self.validate_action_mask(action_mask)

        # Mask out false negatives in the final step
        # if last_step:
        #     false_negative_mask = self.get_false_negative_mask(e_space, e_s, q, e_t, kg)
        #     action_mask *= (1 - false_negative_mask)
        #     self.validate_action_mask(action_mask)

        return (r_space, e_space), action_mask

    # def get_ground_truth_edge_mask(self, e, r_space, e_space, e_s, q, e_t, kg):
    #     ground_truth_edge_mask = ((e == e_s).unsqueeze(1) * (r_space == q.unsqueeze(1)) * (e_space == e_t.unsqueeze(1)))

    #     return ((ground_truth_edge_mask) * (e_s.unsqueeze(1) != kg.dummy_e)).float()

    def get_answer_mask(self, e_space, e_s, q, kg):
        # if kg.args.mask_test_false_negatives:
        answer_vectors = kg.all_target_vectors
        # else:
            # answer_vectors = kg.train_target_vectors
        answer_masks = []
        for i in range(len(e_space)):
            _e_s, _q = int(e_s[i]), int(q[i])
            if not _e_s in answer_vectors or not _q in answer_vectors[_e_s]:
                if self.args.use_gpu:
                    answer_vector = utils.var_cuda(torch.LongTensor([[kg.num_entities]]), args=self.args, use_gpu=True)
                else:
                    answer_vector = utils.var_cuda(torch.LongTensor([[kg.num_entities]]), args=self.args, use_gpu=False)
            else:
                if self.args.use_gpu:
                    answer_vector = answer_vectors[_e_s][_q]
                else:
                    answer_vector = answer_vectors[_e_s][_q].cpu()
            answer_mask = torch.sum(e_space[i].unsqueeze(0) == answer_vector, dim=0).long()
            answer_masks.append(answer_mask)
        answer_mask = torch.cat(answer_masks).view(len(e_space), -1)
        return answer_mask

    # def get_false_negative_mask(self, e_space, e_s, q, e_t, kg):
    #     answer_mask = self.get_answer_mask(e_space, e_s, q, kg)
    #     # This is a trick applied during training where we convert a multi-answer predction problem into several
    #     # single-answer prediction problems. By masking out the other answers in the training set, we are forcing
    #     # the agent to walk towards a particular answer.
    #     # This trick does not affect inference on the test set: at inference time the ground truth answer will not 
    #     # appear in the answer mask. This can be checked by uncommenting the following assertion statement. 
    #     # Note that the assertion statement can trigger in the last batch if you're using a batch_size > 1 since
    #     # we append dummy examples to the last batch to make it the required batch size.
    #     # The assertion statement will also trigger in the dev set inference of NELL-995 since we randomly 
    #     # sampled the dev set from the training data.
    #     # assert(float((answer_mask * (e_space == e_t.unsqueeze(1)).long()).sum()) == 0)
    #     false_negative_mask = (answer_mask * (e_space != e_t.unsqueeze(1)).long()).float()
    #     return false_negative_mask

    def validate_action_mask(self, action_mask):
        action_mask_min = action_mask.min()
        action_mask_max = action_mask.max()
        assert (action_mask_min == 0 or action_mask_min == 1)
        assert (action_mask_max == 0 or action_mask_max == 1)
        assert (not any(action_mask.sum(dim=1) == 0))

    def get_action_embedding(self, action, kg):
        """
        Return (batch) action embedding which is the concatenation of the embeddings of
        the traversed edge and the target node.

        :param action (r, e):
            (Variable:batch) indices of the most recent action
                - r is the most recently traversed edge
                - e is the destination entity.
        :param kg: Knowledge graph enviroment.
        """
        r, e = action
        relation_embedding = kg.get_relation_embeddings(r)
        entity_embedding = kg.get_entity_embeddings(e)
        action_embedding = torch.cat([relation_embedding, entity_embedding], dim=-1)
        return action_embedding

    def define_modules(self):

        input_dim = self.history_dim + self.entity_dim
        self.W1 = nn.Linear(input_dim, self.action_dim)
        self.W2 = nn.Linear(self.action_dim, self.action_dim)
        self.W1Dropout = nn.Dropout(p=self.ff_dropout_rate)
        self.W2Dropout = nn.Dropout(p=self.ff_dropout_rate)
        self.path_encoder = nn.LSTM(input_size=self.action_dim,
                                        hidden_size=self.history_dim,
                                        num_layers=self.history_num_layers,
                                        batch_first=True)

    def initialize_modules(self):
        if self.xavier_initialization:
            nn.init.xavier_uniform_(self.W1.weight)
            nn.init.xavier_uniform_(self.W2.weight)
            for name, param in self.path_encoder.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

class RewardShapingPolicyGradient(PolicyGradient):
    def __init__(self, args, kg, pn, fn):
        super(RewardShapingPolicyGradient, self).__init__(args, kg, pn)
        self.reward_shaping_threshold = args.reward_shaping_threshold

        # Fact network modules
        self.args = args
        self.fn = fn
        self.pre_train_model_embeddings = utils.get_graphsage_embedding(args)
        self.pre_train_model_embeddings.requires_grad = False

        self.fn.model.eval()
        utils.detach_module(self.fn.model)

    def prob(self, e1, pred_e2):
        if self.args.use_gpu is True:
            X = torch.cat([self.pre_train_model_embeddings[e1],self.pre_train_model_embeddings[pred_e2]], dim=1).cuda()
        else:
            X = torch.cat([self.pre_train_model_embeddings[e1],self.pre_train_model_embeddings[pred_e2]], dim=1)
        # return torch.tensor(self.fn.predict_proba(X)[:, 1])
        return torch.tensor((np.argmax(self.fn.predict_proba(X), axis=1) == 1).astype(float) * self.fn.predict_proba(X)[:,1])

    def reward_fun(self, e1, e2, pred_e2):

        if self.args.use_gpu:
            hit_disease_reward = ((pred_e2.cuda().unsqueeze(1) == torch.tensor(self.args.disease_ids).cuda()).sum(dim=1) > 0).float()
            real_reward = self.prob(e1, pred_e2).cuda()
            # real_reward_mask = (real_reward >= self.reward_shaping_threshold).float().cuda()
            # real_reward = real_reward * real_reward_mask * hit_disease_reward
            real_reward[hit_disease_reward == 0] = -1.0 ## if the last node is not disease, set -1.0
            binary_reward = (pred_e2.cuda() == e2.cuda()).float()
            return binary_reward * self.args.tp_reward + (1 - binary_reward) * real_reward
        else:
            hit_disease_reward = ((pred_e2.unsqueeze(1) == torch.tensor(self.args.disease_ids)).sum(dim=1) > 0).float()
            real_reward = self.prob(e1, pred_e2)
            # real_reward_mask = (real_reward >= self.reward_shaping_threshold).float()
            # real_reward = real_reward * real_reward_mask * hit_disease_reward
            real_reward[hit_disease_reward == 0] = -1.0 ## if the last node is not disease, set -1.0
            binary_reward = (pred_e2 == e2).float()
            return binary_reward * self.args.tp_reward + (1 - binary_reward) * real_reward
