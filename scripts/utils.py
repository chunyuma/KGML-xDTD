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
import re
from tqdm import tqdm
# from bmt import Toolkit
from biolink_helper import BiolinkHelper
from neo4j import GraphDatabase
from sklearn.metrics import f1_score
from models import Transition
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

    if concept_pubmed_ids[0] is None or concept_pubmed_ids[1] is None:
        return None

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

def detach_module(mdl):
    for param in mdl.parameters():
        param.requires_grad = False

def get_expert_trans(expert, idx):
    def select_idx(e):
        if isinstance(e, list):
            if not e:
                return []
            return [x[idx] for x in e]
        elif isinstance(e, torch.Tensor):
            return e[idx]
        else:
            return []
    return Transition(*tuple([select_idx(e) for e in list(expert)]))

def load_index(input_path):
    name_to_id, id_to_name = {}, {}
    with open(input_path) as f:
        for index, line in enumerate(f.readlines()):
            name, _ = line.strip().split()
            name_to_id[name] = index
            id_to_name[index] = name
    return name_to_id, id_to_name

def get_depth_of_predicate(predicate_list, biolink_version='2.1.0'):
    # toolkit = Toolkit() for biolink 1.8.1
    # return {predicate:len(toolkit.get_ancestors(predicate)) for predicate in predicate_list}
    biolink_helper = BiolinkHelper(biolink_version=biolink_version)
    return {predicate:len(biolink_helper.get_ancestors(predicate, include_mixins=False)) for predicate in predicate_list}

def hist_to_vocab(_dict):
    return sorted(sorted(_dict.items(), key=lambda x: x[0]), key=lambda x: x[1], reverse=True)

def entity_load_embed(args):
    embedding_folder = os.path.join(args.data_dir, 'kg_init_embeddings')
    embeds = np.load(os.path.join(embedding_folder,'entity_embeddings.npy'))
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

def batch_lookup(M, idx, vector_output=True):
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


def unique_max(unique_x, x, values, args, marker_2D=None, use_gpu=True):
    unique_interval = 2
    unique_values, unique_indices = [], []
    # prevent memory explotion during decoding
    for i in range(0, len(unique_x), unique_interval):
        unique_x_b = unique_x[i:i+unique_interval]
        marker_2D = (unique_x_b.unsqueeze(1) == x.unsqueeze(0)).float()
        values_2D = marker_2D * values.unsqueeze(0) - (1 - marker_2D) * HUGE_INT
        if use_gpu:
            empty_gpu_cache(args)
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

def load_triples(data_path, entity_index_path, relation_index_path, group_examples_by_query=False, seen_entities=None, verbose=False):
    """
    Convert triples stored on disc into indices.
    """
    entity2id, _ = load_index(entity_index_path)
    relation2id, _ = load_index(relation_index_path)

    def triple2ids(source, target, relation):
        return entity2id[source], entity2id[target], relation2id[relation]

    triples = []
    if group_examples_by_query:
        triple_dict = {}
    with open(data_path) as f:
        num_skipped = 0
        for line in f:
            source, target, relation = line.strip().split()
            if seen_entities and (not source in seen_entities or not target in seen_entities):
                num_skipped += 1
                if verbose:
                    print('Skip triple ({}) with unseen entity: {}'.format(num_skipped, line.strip())) 
                continue

            if group_examples_by_query:
                source_id, target_id, relation_id = triple2ids(source, target, relation)
                if source_id not in triple_dict:
                    triple_dict[source_id] = {}
                if relation_id not in triple_dict[source_id]:
                    triple_dict[source_id][relation_id] = set()
                triple_dict[source_id][relation_id].add(target_id)
            else:
                triples.append(triple2ids(source, target, relation))

    if group_examples_by_query:
        for source_id in triple_dict:
            for relation_id in triple_dict[source_id]:
                triples.append((source_id, list(triple_dict[source_id][relation_id]), relation_id))
    print('{} triples loaded from {}'.format(len(triples), data_path))
    return triples

def calculate_f1score(preds, labels, average='binary'):
    preds = np.array(preds)
    y_pred_tags = np.argmax(preds, axis=1)
    abels = np.array(labels)
    f1score = f1_score(labels, y_pred_tags, average=average)
    return f1score
 
def calculate_acc(preds, labels):
    preds = np.array(preds)
    y_pred_tags = np.argmax(preds, axis=1)
    labels = np.array(labels)
    acc = (y_pred_tags == labels).astype(float).mean()
    return acc

def calculate_mrr(drug_disease_pairs, random_pairs, N):
    '''
    This function is used to calculate Mean Reciprocal Rank (MRR)
    reference paper: Knowledge Graph Embedding for Link Prediction: A Comparative Analysis
    '''
    
    ## only use tp pairs
    drug_disease_pairs = drug_disease_pairs.loc[drug_disease_pairs['y']==1,:].reset_index(drop=True)
    
    Q_n = len(drug_disease_pairs)
    score = 0
    for index in range(Q_n):
        query_drug = drug_disease_pairs['source'][index]
        query_disease = drug_disease_pairs['target'][index]
        this_query_score = drug_disease_pairs['prob'][index]
        all_random_probs_for_this_query = list(random_pairs.loc[random_pairs['source'].isin([query_drug]),'prob'])
        all_random_probs_for_this_query += list(random_pairs.loc[random_pairs['target'].isin([query_disease]),'prob'])
        all_random_probs_for_this_query = all_random_probs_for_this_query[:N]
        all_in_list = [this_query_score] + all_random_probs_for_this_query
        rank = list(torch.tensor(all_in_list).sort(descending=True).indices.numpy()).index(0)+1
        score += 1/rank
        
    final_score = score/Q_n
    
    return final_score

def calculate_hitk(drug_disease_pairs, random_pairs, N, k=1):
    '''
    This function is used to calculate Hits@K (H@K)
    reference paper: Knowledge Graph Embedding for Link Prediction: A Comparative Analysis
    '''
    
    ## only use tp pairs
    drug_disease_pairs = drug_disease_pairs.loc[drug_disease_pairs['y']==1,:].reset_index(drop=True)
    
    Q_n = len(drug_disease_pairs)
    count = 0
    for index in range(Q_n):
        query_drug = drug_disease_pairs['source'][index]
        query_disease = drug_disease_pairs['target'][index]
        this_query_score = drug_disease_pairs['prob'][index]
        all_random_probs_for_this_query = list(random_pairs.loc[random_pairs['source'].isin([query_drug]),'prob'])
        all_random_probs_for_this_query += list(random_pairs.loc[random_pairs['target'].isin([query_disease]),'prob'])
        all_random_probs_for_this_query = all_random_probs_for_this_query[:N]
        all_in_list = [this_query_score] + all_random_probs_for_this_query
        rank = list(torch.tensor(all_in_list).sort(descending=True).indices.numpy()).index(0)+1
        if rank <= k:
            count += 1
        
    final_score = count/Q_n
    
    return final_score


def calculate_rank(data_pos_df, all_drug_ids, all_disease_ids, entity_embeddings_dict, all_tp_pairs_dict, fitModel, mode='both'):
    res_dict = dict()
    total = data_pos_df.shape[0]
    for index, (source, target) in enumerate(data_pos_df[['source','target']].to_numpy()):
        print(f"calculating rank {index+1}/{total}", flush=True)
        this_pair = source + '_' + target
        X_drug = np.vstack([np.hstack([entity_embeddings_dict[drug_id],entity_embeddings_dict[target]]) for drug_id in all_drug_ids])
        X_disease = np.vstack([np.hstack([entity_embeddings_dict[source],entity_embeddings_dict[disease_id]]) for disease_id in all_disease_ids])
        all_X = np.concatenate([X_drug,X_disease],axis=0)
        pred_probs = fitModel.predict_proba(all_X)
        temp_df = pd.concat([pd.DataFrame(zip(all_drug_ids,[target]*len(all_drug_ids))),pd.DataFrame(zip([source]*len(all_disease_ids),all_disease_ids))]).reset_index(drop=True)
        temp_df[2] = temp_df[0] + '_' + temp_df[1]
        temp_df[3] = pred_probs[:,1]
        this_row = temp_df.loc[temp_df[2]==this_pair,:].reset_index(drop=True).iloc[[0]]
        temp_df = temp_df.loc[temp_df[2]!=this_pair,:].reset_index(drop=True)
        if mode == 'both':
            ## without filter
            # (1) for drug
            temp_df_1 = pd.concat([temp_df.loc[temp_df[1] == target,:],this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            w_drug_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            # (2) for disease
            temp_df_1 = pd.concat([temp_df.loc[temp_df[0] == source,:],this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            w_disease_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            # (3) for both 
            temp_df_1 = pd.concat([temp_df,this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            w_both_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            ## filter 
            temp_df = temp_df.loc[~temp_df[2].isin(list(all_tp_pairs_dict.keys())),:].reset_index(drop=True)
             # (1) for drug
            temp_df_1 = pd.concat([temp_df.loc[temp_df[1] == target,:],this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            drug_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            # (2) for disease
            temp_df_1 = pd.concat([temp_df.loc[temp_df[0] == source,:],this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            disease_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            # (3) for both 
            temp_df_1 = pd.concat([temp_df,this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            both_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            res_dict[(source, target)] = [(w_drug_rank, w_disease_rank, w_both_rank), (drug_rank, disease_rank, both_rank)]

        elif mode == 'filter':
            ## filter 
            temp_df = temp_df.loc[~temp_df[2].isin(list(all_tp_pairs_dict.keys())),:].reset_index(drop=True)
             # (1) for drug
            temp_df_1 = pd.concat([temp_df.loc[temp_df[1] == target,:],this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            drug_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            # (2) for disease
            temp_df_1 = pd.concat([temp_df.loc[temp_df[0] == source,:],this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            disease_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            # (3) for both 
            temp_df_1 = pd.concat([temp_df,this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            both_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            res_dict[(source, target)] = [(None, None, None), (drug_rank, disease_rank, both_rank)] 
        else:
            ## without filter
            # (1) for drug
            temp_df_1 = pd.concat([temp_df.loc[temp_df[1] == target,:],this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            w_drug_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            # (2) for disease
            temp_df_1 = pd.concat([temp_df.loc[temp_df[0] == source,:],this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            w_disease_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            # (3) for both 
            temp_df_1 = pd.concat([temp_df,this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            w_both_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            res_dict[(source, target)] = [(w_drug_rank, w_disease_rank, w_both_rank), (None, None, None)]
    return res_dict


def plot_cutoff(dfs, plot_title, outfile_path, class_label=[1,0,2], title_post = ["TP pairs", "TN pairs", "Random pairs"]):

    color = ["xkcd:dark magenta","xkcd:dark turquoise","xkcd:azure","xkcd:purple blue","xkcd:scarlet",
        "xkcd:orchid", "xkcd:pumpkin", "xkcd:gold", "xkcd:peach", "xkcd:neon green", "xkcd:grey blue"]

    cutoff_n_list = []

    for df in dfs:
        cutoffs = [x/100 for x in range(101)]
        cutoff_n_list += [[df["treat_prob"][df["treat_prob"] >= cutoff].count()/len(df) for cutoff in cutoffs]]

    tp_to_tn_diff = [cutoff_n_list[class_label.index(1)][index] - cutoff_n_list[class_label.index(0)][index] for index in range(101)]
    tp_to_random_diff = [cutoff_n_list[class_label.index(1)][index] - cutoff_n_list[class_label.index(2)][index] for index in range(101)]
    min_val = [min(tp_to_tn_diff[index],tp_to_random_diff[index]) for index in range(101)]

    fig, ax1 = plt.subplots()
    for index, label in enumerate(class_label):
        ax1.plot(cutoffs, cutoff_n_list[class_label.index(label)], color=color[index], label=title_post[index])
    ax1.set_ylabel("Rate of Postitive Predictions",fontsize=12)
    ax1.set_xlabel("Probability Cutoff",fontsize=12)
    ax1.legend(loc="upper right")
    ax2=ax1.twinx()
    ax2.plot(cutoffs, min_val, color=color[index+1], ls = ':')
    ax2.set_ylabel("Min RPP Difference",color=color[index+1],fontsize=12)
    plt.title(plot_title)
    plt.savefig(outfile_path)
    plt.close()


def beam_search(args, source_ids, env, model):
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
            unique_action_weighted_prob_dist, unique_idx = unique_max(unique_e_space_b, e_space_b, action_weighted_prob_dist_b, args=args, use_gpu=args.use_gpu)
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
        next_r = pad_and_cat(next_r_list, padding_value=env.kg.dummy_r).view(-1)
        next_e = pad_and_cat(next_e_list, padding_value=env.kg.dummy_e).view(-1)
        action_prob = pad_and_cat(action_prob_list, padding_value=0)
        action_offset = pad_and_cat(action_offset_list, padding_value=-1)
        return (next_r, next_e), action_prob.view(-1), action_offset.view(-1)

    # Initialization
    r_s = int_var_cuda(source_ids, args=args, use_gpu=False)
    batch_size = len(r_s)
    env.initialize_path(r_s)
    num_steps = args.max_path 


    if args.use_gpu:
        action_log_weighted_prob = zeros_var_cuda(batch_size, args=args, use_gpu=True)
    else:
        action_log_weighted_prob = zeros_var_cuda(batch_size, args=args, use_gpu=False)


    print(f"", flush=True)
    for t in range(num_steps):
        print(f"Here is step {t+1} for beam search", flush=True)

        state_inputs = model.process_state(model.history_len, env._batch_curr_state)
        probs, _ = model.policy_net(state_inputs.to(args.device), env._batch_curr_action_spaces)
        if args.use_gpu:
            empty_gpu_cache(args)
        weighted_prob = action_log_weighted_prob.view(-1, 1) + args.factor**t * torch.log((probs+TINY_VALUE) * torch.count_nonzero(probs, dim=1).view(-1,1))
        if t == num_steps - 1:
            action, action_log_weighted_prob, action_offset = top_k_answer_unique(weighted_prob, env._batch_curr_action_spaces, batch_size, args.topk)
        else:
            action, action_log_weighted_prob, action_offset = top_k_action(weighted_prob, env._batch_curr_action_spaces, batch_size, args.topk)
        if args.use_gpu:
            empty_gpu_cache(args)
        env.batch_step(action, offset=action_offset)
        if args.use_gpu:
            empty_gpu_cache(args)

    output_beam_size = int(action[0].size()[0] / batch_size)
    beam_search_output = dict()
    beam_search_output['paths'] = env._batch_path
    beam_search_output['pred_prob_scores'] = action_log_weighted_prob.cpu()
    beam_search_output['output_beam_size'] = output_beam_size
    env.reset()

    return beam_search_output

def evaluate(args, drug_disease_dict, env, model, all_drug_disease_dict, save_paths=False):

    model.policy_net.eval()
    eval_drugs = torch.tensor(list(drug_disease_dict.keys()))
    args.logger.info('Evaluating model')
    with torch.no_grad():
        ## predict paths for drugs
        eval_dataloader = ACDataLoader(list(range(len(eval_drugs))), args.eval_batch_size)
        pbar = tqdm(total=eval_dataloader.num_paths)
        if save_paths:
            all_paths_r, all_paths_e, all_prob_scores = [], [], []
        pred_diseases = dict()
        while eval_dataloader.has_next():
            dids_idx = eval_dataloader.get_batch()
            source_ids = eval_drugs[dids_idx]
            res = beam_search(args, source_ids, env, model)
            if args.use_gpu:
                empty_gpu_cache(args)
            if save_paths:
                all_paths_r += [res['paths'][0]]
                all_paths_e += [res['paths'][1]]
                all_prob_scores += [res['pred_prob_scores']]
            for index in range(0,res['paths'][1].shape[0],args.topk):
                drug_id = int(res['paths'][1][index][0])
                pred_list = res['paths'][1][index:(index+args.topk),-1].tolist()
                disease_list = list(set(pred_list).intersection(set(args.disease_ids)))
                pred_diseases[drug_id] = dict()
                pred_diseases[drug_id]['list'] = disease_list
                if len(disease_list) !=0:
                    pred_diseases[drug_id]['pred_score'] = env.prob([drug_id]*len(disease_list),disease_list)
                else:
                    pred_diseases[drug_id]['pred_score'] = torch.tensor([])
            pbar.update(len(source_ids))
    
        failed_pred = 0
        avg_pred_score = []
        recalls, precisions, hits = [], [], []
        all_recalls, all_precisions, all_hits = [], [], []
        hit_disease_target_pairs = dict()
        for drug_id in drug_disease_dict.keys():
            if len(pred_diseases[drug_id]['list']) == 0:
                failed_pred += 1
                continue
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

        args.logger.info(f'{failed_pred}/{len(drug_disease_dict.keys())} from evaluation dataset have no disease prediction')
        avg_pred_score = np.mean(avg_pred_score)
        avg_recall = np.mean(recalls) * 100
        avg_precision = np.mean(precisions) * 100
        avg_hit = np.mean(hits) * 100
        args.logger.info(f'Avg prediction score={avg_pred_score:.3f}')
        args.logger.info(f'Evaluation dataset only: Recall={avg_recall:.3f} | HR={avg_hit:.3f} | Precision={avg_precision:.3f}')
        all_avg_recall = np.mean(all_recalls) * 100
        all_avg_precision = np.mean(all_precisions) * 100
        all_avg_hit = np.mean(all_hits) * 100
        args.logger.info(f'all datasets (train, val and test): Recall={all_avg_recall:.3f} | HR={all_avg_hit:.3f} | Precision={all_avg_precision:.3f}')

        if save_paths:
            all_paths_r = torch.cat(all_paths_r)
            all_paths_e = torch.cat(all_paths_e)
            all_prob_scores = torch.cat(all_prob_scores)
            return {'paths': [all_paths_r,all_paths_e], 'prob_scores': all_prob_scores, 'hit_pairs': hit_disease_target_pairs}
        else:
            return None