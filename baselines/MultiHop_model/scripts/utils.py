import numpy as np
import os, sys
import logging
import logging.handlers
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import math
from tqdm import tqdm
# from bmt import Toolkit
from neo4j import GraphDatabase
from sklearn.metrics import f1_score
plt.switch_backend('agg')

SELF_LOOP_RELATION = 'SELF_LOOP_RELATION'
DUMMY_RELATION = 'DUMMY_RELATION'
DUMMY_ENTITY = 'DUMMY_ENTITY'

DUMMY_RELATION_ID = 0
SELF_LOOP_RELATION_ID = 1
DUMMY_ENTITY_ID = 0
EPSILON = float(np.finfo(float).eps)
HUGE_INT = 1e31
TINY_VALUE = 1e-41
NGD_normalizer = 2.2e+7 * 20  # From PubMed home page there are 27 million articles; avg 20 MeSH terms per article

class ACDataLoader(object):
    def __init__(self, indexes, batch_size, permutation=True):
        self.indexes = np.array(indexes)
        self.num_paths = len(indexes)
        self.batch_size = batch_size
        self._permutation = permutation
        self.reset()

    def reset(self):
        if self._permutation:
            self._rand_perm = np.random.permutation(self.num_paths)
        else:
            self._rand_perm = np.array(range(self.num_paths))
        self._start_idx = 0
        self._has_next = True

    def has_next(self):
        return self._has_next

    def get_batch(self):
        if not self._has_next:
            return None
        # Multiple users per batch
        end_idx = min(self._start_idx + self.batch_size, self.num_paths)
        batch_idx = self._rand_perm[self._start_idx:end_idx]
        batch_indexes = self.indexes[batch_idx]
        self._has_next = self._has_next and end_idx < self.num_paths
        self._start_idx = end_idx
        return batch_indexes.tolist()

class Neo4jConnection:

    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            print("Failed to create the driver:", e)

    def close(self):
        if self.__driver is not None:
            self.__driver.close()

    def query(self, query, db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try:
            session = self.__driver.session(database=db) if db is not None else self.__driver.session()
            response = pd.DataFrame(session.run(query))
        except Exception as e:
            print("Query failed:", e)
        finally:
            if session is not None:
                session.close()
        return response

def calculate_ngd(concept_pubmed_ids):

    marginal_counts = list(map(lambda pmid_list: len(set(pmid_list)), concept_pubmed_ids))
    joint_count = len(set(concept_pubmed_ids[0]).intersection(set(concept_pubmed_ids[1])))

    if 0 in marginal_counts or 0. in marginal_counts:
        return None
    elif joint_count == 0 or joint_count == 0.:
        return None
    else:
        try:
            return (max([math.log(count) for count in marginal_counts]) - math.log(joint_count)) / \
                (math.log(NGD_normalizer) - min([math.log(count) for count in marginal_counts]))
        except ValueError:
            return None

def clean_up_desc(string):
    if type(string) is str:
        # Removes all of the "UMLS Semantic Type: UMLS_STY:XXXX;" bits from descriptions
        string = re.sub("UMLS Semantic Type: UMLS_STY:[a-zA-Z][0-9]{3}[;]?", "", string).strip().strip(";")
        if string == 'None':
            return ''
        elif len(re.findall("^COMMENTS: ", string)) != 0:
            return re.sub("^COMMENTS: ","", string)
        elif len(re.findall("-!- FUNCTION: ", string)) != 0:
            part1 = [part for part in string.split('-!-') if len(re.findall("^ FUNCTION: ", part)) != 0][0].replace(' FUNCTION: ','')
            part2 = re.sub(' \{ECO:.*\}.','',re.sub(" \(PubMed:[0-9]*,? ?(PubMed:[0-9]*,?)?\)","",part1))
            return part2
        elif len(re.findall("Check for \"https:\/\/www\.cancer\.gov\/", string)) != 0:
            return re.sub("Check for \"https:\/\/www\.cancer\.gov\/.*\" active clinical trials using this agent. \(\".*NCI Thesaurus\); ","",string)
        else:
            return string
    elif string is None:
        return ''
    else:
        raise ValueError('Not expected type {type(string)}')

def clean_up_name(string):
    if type(string) is str:
        if string == 'None':
            return ''
        else:
            return string
    elif string is None:
        return ''
    else:
        raise ValueError('Not expected type {type(string)}')

def entity_load_embed(args):
    embedding_folder = os.path.join(args.data_dir, 'kg_init_embeddings')
    embeds = np.load(os.path.join(embedding_folder,'entity_embeddings.npy'))
    # embedding_file_folder = os.path.dirname(args.pretrain_model_path)
    # embeds = np.load(os.path.join(embedding_file_folder, 'entity_embeddings.npy'))[1:,:]
    return torch.tensor(embeds).type(torch.float)

def get_graphsage_embedding(args):
    embedding_file_folder = os.path.dirname(args.pretrain_model_path)
    with open(os.path.join(embedding_file_folder, 'entity_embeddings.npy'),'rb') as infile:
        entity_embeddings = np.load(infile)
    return torch.tensor(entity_embeddings).type(torch.float)

def relation_load_embed(args):
    embedding_folder = os.path.join(args.data_dir, 'kg_init_embeddings')
    embeds = np.load(os.path.join(embedding_folder,'relation_embeddings.npy'))
    return torch.tensor(embeds).type(torch.float)

def entity_type_load_embed(args):
    embedding_folder = os.path.join(args.data_dir, 'kg_init_embeddings')
    embeds = np.load(os.path.join(embedding_folder,'entity_type_embeddings.npy'))
    return torch.tensor(embeds).type(torch.float)

def load_index(input_path):
    name_to_id, id_to_name = {}, {}
    with open(input_path) as f:
        for index, line in enumerate(f.readlines()):
            name, _ = line.strip().split()
            name_to_id[name] = index
            id_to_name[index] = name
    return name_to_id, id_to_name

def batch_lookup(M, idx, vector_output=True):
    """
    Perform batch lookup on matrix M using indices idx.
    :param M: (Variable) [batch_size, seq_len] Each row of M is an independent population.
    :param idx: (Variable) [batch_size, sample_size] Each row of idx is a list of sample indices.
    :param vector_output: If set, return a 1-D vector when sample size is 1.
    :return samples: [batch_size, sample_size] samples[i, j] = M[idx[i, j]]
    """
    batch_size, w = M.size()
    batch_size2, sample_size = idx.size()
    assert(batch_size == batch_size2)

    if sample_size == 1 and vector_output:
        samples = torch.gather(M, 1, idx).view(-1)
    else:
        samples = torch.gather(M, 1, idx)
    return samples

def empty_gpu_cache(args):
    with torch.cuda.device(f'cuda:{args.gpu}'):
        torch.cuda.empty_cache()

def convert_to_dist(x):
    x += EPSILON
    return x / x.sum(1, keepdim=True)


def detach_module(mdl):
    for param in mdl.parameters():
        param.requires_grad = False

def entropy(p):
    return torch.sum(-p * safe_log(p), 1)


def weighted_softmax(v, w, dim=-1):
    exp_v = torch.exp(v)
    weighted_exp_v = w * exp_v
    return weighted_exp_v / torch.sum(weighted_exp_v, dim, keepdim=True)


def format_triple(triple, kg):
    e1, e2, r = triple
    rel = kg.id2relation[r] if r != kg.self_edge else '<null>'
    if not rel.endswith('_inv'):
        return '{} -{}-> {}'.format(
            kg.id2entity[e1], rel, kg.id2entity[e2])
    else:
        return '{} <-{}- {}'.format(
            kg.id2entity[e1], rel, kg.id2entity[e2])


def format_path(path_trace, kg):
    def get_most_recent_relation(j):
        relation_id = int(path_trace[j][0])
        if relation_id == kg.self_edge:
            return 'self_loop'
        else:
            return kg.id2relation[relation_id]

    def get_most_recent_entity(j):
        return kg.id2entity[int(path_trace[j][1])]

    path_str = get_most_recent_entity(0)
    for j in range(1, len(path_trace)):
        rel = get_most_recent_relation(j)
        if not rel.endswith('_inv'):
            path_str += ' -{}-> '.format(rel)
        else:
            path_str += ' <-{}- '.format(rel[:-4])
        path_str += get_most_recent_entity(j)
    return path_str


def format_rule(rule, kg):
    rule_str = ''
    for j in range(len(rule)):
        relation_id = int(rule[j])
        rel = kg.id2relation[relation_id]
        if not rel.endswith('_inv'):
            rule_str += '-{}-> '.format(rel)
        else:
            rule_str += '<-{}-'.format(rel)
    return rule_str

def ones_var_cuda(s, args, requires_grad=False, use_gpu=True):
    if use_gpu is True:
        return Variable(torch.ones(s), requires_grad=requires_grad).to(args.device)
    else:
        return Variable(torch.ones(s), requires_grad=requires_grad).long()

def zeros_var_cuda(s, args, requires_grad=False, use_gpu=True):
    if use_gpu is True:
        return Variable(torch.zeros(s), requires_grad=requires_grad).to(args.device)
    else:
        return Variable(torch.zeros(s), requires_grad=requires_grad).long()

def int_var_cuda(x, args, requires_grad=False, use_gpu=True):
    if use_gpu is True:
        return Variable(x, requires_grad=requires_grad).long().to(args.device)
    else:
        return Variable(x, requires_grad=requires_grad).long()

def var_cuda(x, args, requires_grad=False, use_gpu=True):
    if use_gpu is True:
        return Variable(x, requires_grad=requires_grad).to(args.device)
    else:
        return Variable(x, requires_grad=requires_grad).long()

def var_to_numpy(x):
    return x.data.cpu().numpy()

def pad_and_cat(a, padding_value, padding_dim=1):
    max_dim_size = max([x.size()[padding_dim] for x in a])
    padded_a = []
    for x in a:
        if x.size()[padding_dim] < max_dim_size:
            res_len = max_dim_size - x.size()[1]
            pad = nn.ConstantPad1d((0, res_len), padding_value)
            padded_a.append(pad(x))
        else:
            padded_a.append(x)
    return torch.cat(padded_a, dim=0)

def rearrange_vector_list(l, offset):
    for i, v in enumerate(l):
        l[i] = v[offset]

def safe_log(x):
    return torch.log(x + EPSILON)


def tile_along_beam(v, beam_size, dim=0):
    """
    Tile a tensor along a specified dimension for the specified beam size.
    :param v: Input tensor.
    :param beam_size: Beam size.
    """
    if dim == -1:
        dim = len(v.size()) - 1
    v = v.unsqueeze(dim + 1)
    v = torch.cat([v] * beam_size, dim=dim+1)
    new_size = []
    for i, d in enumerate(v.size()):
        if i == dim + 1:
            new_size[-1] *= d
        else:
            new_size.append(d)
    return v.view(new_size)


# Flatten and pack nested lists using recursion
def flatten(l):
    flatten_l = []
    for c in l:
        if type(c) is list or type(c) is tuple:
            flatten_l.extend(flatten(c))
        else:
            flatten_l.append(c)
    return flatten_l


def pack(l, a):
    """
    Pack a flattened list l into the structure of the nested list a.
    """
    nested_l = []
    for c in a:
        if type(c) is not list:
            nested_l.insert(l[0], 0)
            l.pop(0)


def unique_max(unique_x, x, values, marker_2D=None, use_gpu=True):
    unique_interval = 2
    unique_values, unique_indices = [], []
    # prevent memory explotion during decoding
    for i in range(0, len(unique_x), unique_interval):
        unique_x_b = unique_x[i:i+unique_interval]
        marker_2D = (unique_x_b.unsqueeze(1) == x.unsqueeze(0)).float()
        values_2D = marker_2D * values.unsqueeze(0) - (1 - marker_2D) * HUGE_INT
        if use_gpu:
             torch.cuda.empty_cache()
        unique_values_b, unique_idx_b = values_2D.max(dim=1)
        unique_values.append(unique_values_b)
        unique_indices.append(unique_idx_b)
    unique_values = torch.cat(unique_values)
    unique_idx = torch.cat(unique_indices)
    return unique_values, unique_idx


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_logger(logname):
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s  [%(levelname)s]  %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def compute_mrr(examples, scores, all_answers, beam_size=5000, is_batch=False, verbose=False):
    """
    Compute MRR.
    """
    assert (len(examples) == scores.shape[0])
    # mask all other true positive targets and dummy targets
    dummy_mask = [DUMMY_ENTITY_ID, SELF_LOOP_ENTITY_ID]
    for i, example in enumerate(examples):
        source, target, relation = example
        target_multi = dummy_mask + list(all_answers[source][relation]) 
        target_score = float(scores[i, target])
        scores[i, target_multi] = 0
        scores[i, target] = target_score
    
    # sort and rank
    top_k_scores, top_k_targets = torch.topk(scores, min(scores.size(1), beam_size))
    top_k_targets = top_k_targets.cpu().numpy()

    mrr = 0
    for i, example in enumerate(examples):
        _, target, _ = example
        pos = np.where(top_k_targets[i] == target)[0]
        if len(pos) > 0:
            pos = pos[0]
            mrr += 1.0 / (pos + 1)

    if is_batch is False:
        mrr = float(mrr) / len(examples)

        if verbose:
            print('MRR = {:.3f}'.format(mrr))
    else:
        mrr = mrr

    return mrr


def hits_at_k(k_list, examples, scores, all_answers, beam_size=5000, is_batch=False, verbose=False):
    """
    Compute Hits@K.
    """
    assert(len(k_list) > 0)
    assert(len(examples) == scores.shape[0])
    # mask all other true positive targets and dummy targets
    dummy_mask = [DUMMY_ENTITY_ID, SELF_LOOP_ENTITY_ID]
    for i, example in enumerate(examples):
        source, target, relation = example
        target_multi = dummy_mask + list(all_answers[source][relation])
        target_score = float(scores[i, target])
        scores[i, target_multi] = 0
        scores[i, target] = target_score
        
    # sort and rank
    top_k_scores, top_k_targets = torch.topk(scores, min(scores.size(1), beam_size))
    top_k_targets = top_k_targets.cpu().numpy()

    hits_at_k_list = [0] * len(k_list)
    for i, example in enumerate(examples):
        source, target, relation = example
        pos = np.where(top_k_targets[i] == target)[0]
        if len(pos) > 0:
            pos = pos[0]
            for index, k in enumerate(k_list):
                if pos < k:
                    hits_at_k_list[index] += 1

    if is_batch is False:
        for index, hits_at_k in enumerate(hits_at_k_list):
            hits_at_k = float(hits_at_k) / len(examples)
            hits_at_k_list[index] = hits_at_k
        
        if verbose:
            for index, k in enumerate(k_list):
                print(f'Hits@{k} = {hits_at_k_list[index]:.3f}')
    else:
        for index, hits_at_k in enumerate(hits_at_k_list):
            hits_at_k_list[index] = hits_at_k

    return hits_at_k_list

def calculate_f1score(preds, labels, threshold=0.5):
    preds = np.array(preds)
    labels = np.array(labels)
    preds = (np.array(preds)>threshold).astype(int)
    f1score = f1_score(labels, preds, average='binary')
    return f1score

def calculate_acc(preds, labels, threshold=0.5):
    preds = np.array(preds)
    labels = np.array(labels)
    preds = (np.array(preds)>threshold).astype(int)
    acc = (preds == labels).astype(float).mean()
    return acc

def beam_search(args, pn, e_s, kg, num_steps, beam_size):
    """
    Beam search from source.
    """

    def top_k_action(action_weighted_prob_dist, action_space, batch_size, beam_size):
        """
        Get top k actions.
        """
        full_size = len(action_weighted_prob_dist)
        assert (full_size % batch_size == 0)
        last_k = int(full_size / batch_size)
        (r_space, e_space), _ = action_space
        action_space_size = r_space.size()[1]
        action_weighted_prob_dist = action_weighted_prob_dist.view(batch_size, -1)
        beam_action_space_size = action_weighted_prob_dist.size()[1]
        k = min(beam_size, beam_action_space_size)
        action_weighted_prob, action_ind = torch.topk(action_weighted_prob_dist, k)
        next_r = batch_lookup(r_space.view(batch_size, -1), action_ind).view(-1)
        next_e = batch_lookup(e_space.view(batch_size, -1), action_ind).view(-1)
        action_weighted_prob = action_weighted_prob.view(-1)
        action_beam_offset = torch.div(action_ind, action_space_size, rounding_mode='floor')
        if args.use_gpu:
            action_batch_offset = int_var_cuda(torch.arange(batch_size) * last_k, args=args, use_gpu=True).unsqueeze(1)
        else:
            action_batch_offset = int_var_cuda(torch.arange(batch_size) * last_k, args=args, use_gpu=False).unsqueeze(1)
        action_offset = (action_batch_offset + action_beam_offset).view(-1)
        return (next_r, next_e), action_weighted_prob, action_offset

    def top_k_answer_unique(action_weighted_prob_dist, action_space, batch_size, beam_size):
        """
        Get top k unique entities
        """
        full_size = len(action_weighted_prob_dist)
        assert (full_size % batch_size == 0)
        last_k = int(full_size / batch_size)
        (r_space, e_space), _ = action_space
        action_space_size = r_space.size()[1]

        r_space = r_space.view(batch_size, -1)
        e_space = e_space.view(batch_size, -1)
        action_weighted_prob_dist = action_weighted_prob_dist.view(batch_size, -1)
        beam_action_space_size = action_weighted_prob_dist.size()[1]
        assert (beam_action_space_size % action_space_size == 0)
        k = min(beam_size, beam_action_space_size)
        next_r_list, next_e_list = [], []
        action_prob_list = []
        action_offset_list = []
        for i in range(batch_size):
            action_weighted_prob_dist_b = action_weighted_prob_dist[i]
            r_space_b = r_space[i]
            e_space_b = e_space[i]
            if args.use_gpu:
                unique_e_space_b = var_cuda(torch.unique(e_space_b.data.cpu()), args=args, use_gpu=True)
            else:
                unique_e_space_b = var_cuda(torch.unique(e_space_b.data.cpu()), args=args, use_gpu=False)
            unique_action_weighted_prob_dist, unique_idx = unique_max(unique_e_space_b, e_space_b, action_weighted_prob_dist_b, use_gpu=args.use_gpu)
            k_prime = min(len(unique_e_space_b), k)
            top_unique_action_weighted_prob_dist, top_unique_idx2 = torch.topk(unique_action_weighted_prob_dist, k_prime)
            top_unique_idx = unique_idx[top_unique_idx2]
            top_unique_beam_offset = torch.div(top_unique_idx, action_space_size, rounding_mode='floor')
            top_r = r_space_b[top_unique_idx]
            top_e = e_space_b[top_unique_idx]
            next_r_list.append(top_r.unsqueeze(0))
            next_e_list.append(top_e.unsqueeze(0))
            action_prob_list.append(top_unique_action_weighted_prob_dist.unsqueeze(0))
            top_unique_batch_offset = i * last_k
            top_unique_action_offset = top_unique_batch_offset + top_unique_beam_offset
            action_offset_list.append(top_unique_action_offset.unsqueeze(0))
        next_r = pad_and_cat(next_r_list, padding_value=kg.dummy_r).view(-1)
        next_e = pad_and_cat(next_e_list, padding_value=kg.dummy_e).view(-1)
        action_prob = pad_and_cat(action_prob_list, padding_value=0)
        action_offset = pad_and_cat(action_offset_list, padding_value=-1)
        return (next_r, next_e), action_prob.view(-1), action_offset.view(-1)

    def adjust_search_trace(search_trace, action_offset):
        for i, (r, e) in enumerate(search_trace):
            new_r = r[action_offset]
            new_e = e[action_offset]
            search_trace[i] = (new_r, new_e)

    assert (num_steps >= 1)
    batch_size = len(e_s)

    # Initialization
    if args.use_gpu:
        r_s = int_fill_var_cuda(e_s.size(), kg.self_edge, args=args, use_gpu=True)
    else:
        r_s = int_fill_var_cuda(e_s.size(), kg.self_edge, args=args, use_gpu=False)
    seen_nodes = e_s.unsqueeze(1)
    init_action = (r_s, e_s)
    # path encoder
    pn.initialize_path(init_action, kg)
    search_trace = [(r_s, e_s)]

    # Run beam search for num_steps
    # [batch_size*k], k=1
    if args.use_gpu:
        action_log_weighted_prob = zeros_var_cuda(batch_size, args=args, use_gpu=True)
    else:
        action_log_weighted_prob = zeros_var_cuda(batch_size, args=args, use_gpu=False)


    action = init_action
    print(f"", flush=True)
    for t in range(num_steps):
        print(f"Here is step {t+1} for beam search", flush=True)
        last_r, e = action
        assert(e.size()[0] % batch_size == 0)
        k = int(e.size()[0] / batch_size)
        e_s = tile_along_beam(e_s.view(batch_size, -1)[:, 0], k)
        obs = [e_s, t==(num_steps-1), last_r, seen_nodes]
        # one step forward in search
        db_outcomes, _, _ = pn.transit(e, obs, kg, merge_aspace_batching_outcome=True)
        if args.use_gpu:
            torch.cuda.empty_cache()
        action_space, action_dist = db_outcomes[0]
        # => [batch_size*k, action_space_size]
        # log_action_dist = log_action_prob.view(-1, 1) + safe_log(action_dist)
        weighted_prob = action_log_weighted_prob.view(-1, 1) + args.factor**t * torch.log((action_dist+TINY_VALUE) * torch.count_nonzero(action_dist, dim=1).view(-1,1))
        # [batch_size*k, action_space_size] => [batch_size*new_k]
        if t == num_steps - 1:
            action, action_log_weighted_prob, action_offset = top_k_answer_unique(weighted_prob, action_space, batch_size, beam_size)
        else:
            action, action_log_weighted_prob, action_offset = top_k_action(weighted_prob, action_space, batch_size, beam_size)
        if args.use_gpu:
            torch.cuda.empty_cache()
        # rearrange_vector_list(log_action_probs, action_offset)
        # log_action_probs.append(action_log_weighted_prob)
        pn.update_path(action, kg, offset=action_offset)
        if args.use_gpu:
            seen_nodes = torch.cat([seen_nodes[action_offset], action[1].unsqueeze(1)], dim=1)
        else:
            seen_nodes = torch.cat([seen_nodes[action_offset], action[1].unsqueeze(1).cpu()], dim=1)
        adjust_search_trace(search_trace, action_offset)
        search_trace.append(action)
        if args.use_gpu:
            torch.cuda.empty_cache()

    output_beam_size = int(action[0].size()[0] / batch_size)
    # [batch_size*beam_size] => [batch_size, beam_size]
    beam_search_output = dict()
    # beam_search_output['pred_e2s'] = action[1].view(batch_size, -1)
    beam_search_output['pred_prob_scores'] = action_log_weighted_prob.cpu()
    # beam_search_output['log_action_probs'] = log_action_probs
    beam_search_output['output_beam_size'] = output_beam_size
    beam_search_output['paths'] = [torch.vstack(x).T for x in zip(*search_trace)]

    return beam_search_output