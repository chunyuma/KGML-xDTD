import os, sys
path_list = os.getcwd().split('/')
index = path_list.index('KGML-xDTD')
script_path = '/'.join(path_list[:(index+1)] + ['scripts'])
sys.path.append(script_path)
from utils import ACDataLoader, empty_gpu_cache, set_random_seed, load_index, pad_and_cat, entity_load_embed, relation_load_embed, ones_var_cuda, zeros_var_cuda, int_var_cuda, var_cuda, NodeSynonymizer

import pickle
import joblib
import torch
from torch.utils.data import Dataset
import logging
import graph_tool.all as gt
from knowledge_graph import KnowledgeGraph
from kg_env import KGEnvironment
from models import DiscriminatorActorCritic

nodesynonymizer = NodeSynonymizer()

DUMMY_RELATION_ID = 0
SELF_LOOP_RELATION_ID = 1
DUMMY_ENTITY_ID = 0
TINY_VALUE = 1e-41

def load_graphsage_unsupervised_embeddings(data_path: str, use_node_attribute: bool = True):
    if use_node_attribute:
        file_path = os.path.join(data_path, 'entity_embeddings', 'unsuprvised_graphsage_entity_embeddings.pkl')
    else:
        file_path = os.path.join(data_path, 'entity_embeddings', 'unsuprvised_graphsage_entity_embeddings_wo_node_attributes.pkl')
    with open(file_path, 'rb') as infile:
        entity_embeddings_dict = pickle.load(infile)
    return entity_embeddings_dict

def load_biobert_embeddings(data_path: str):
    file_path = os.path.join(data_path, 'entity_embeddings', 'embedding_biobert_namecat.pkl')
    with open(file_path,'rb') as infile:
        biobert_embeddings_dict = pickle.load(infile)
    return biobert_embeddings_dict

def load_drp_module(model_path: str):
    file_path = os.path.join(model_path,'kgml_xdtd','drp_module','model.pt')
    fitModel = joblib.load(file_path)
    return fitModel

def load_moa_module(args, model_path: str):
    kg = KnowledgeGraph(args, bandwidth=args.bandwidth, emb_dropout_rate=args.emb_dropout_rate, bucket_interval=args.bucket_interval, load_graph=True)
    if args.use_gpu:
        empty_gpu_cache(args)

    env = KGEnvironment(args, kg, max_path_len=args.max_path, state_pre_history=args.state_history)
    fitModel = DiscriminatorActorCritic(args, kg, args.state_history, args.gamma, args.target_update, args.ac_hidden, args.disc_hidden, args.metadisc_hidden)
    args.policy_net_file = os.path.join(model_path,'kgml_xdtd','moa_module','model.pt')
    policy_net = torch.load(args.policy_net_file, map_location=args.device)
    model_temp = fitModel.policy_net.state_dict()
    model_temp.update(policy_net)
    fitModel.policy_net.load_state_dict(model_temp)
    del policy_net
    del model_temp
    if args.use_gpu:
        empty_gpu_cache(args)
    return kg, env, fitModel

def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s  [%(levelname)s]  %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def check_device(logger, use_gpu: bool = False, gpu: int = 0):
    if use_gpu and torch.cuda.is_available():
        use_gpu = True
        device = torch.device(f'cuda:{gpu}')
        torch.cuda.set_device(gpu)
    elif use_gpu:
        logger.info('No GPU is detected in this computer. Use CPU instead.')
        use_gpu = False
        device = 'cpu'
    else:
        use_gpu = False
        device = 'cpu'

    return [use_gpu, device]

def check_curie(curie: str, entity2id):
    if curie is None:
        return (None, None)
    res = nodesynonymizer.get_canonical_curies(curie)[curie]
    if res is not None:
        preferred_curie = nodesynonymizer.get_canonical_curies(curie)[curie]['preferred_curie']
    else:
        preferred_curie = None
    if preferred_curie in entity2id:
        return (preferred_curie, entity2id[preferred_curie])
    else:
        return (preferred_curie, None)

def check_curie_available(logger, curie: str, available_curies_dict: dict):
    normalized_result = nodesynonymizer.get_canonical_curies(curie)[curie]
    if normalized_result:
        curie = normalized_result['preferred_curie']
    else:
        curie = curie
    
    if curie in available_curies_dict:
        return [True, curie]
    else:
        return [False, None]

def load_gt_kg(kg):
    G = gt.Graph()
    kg_tmp = dict()
    for source in kg.graph:
        for (relation, target) in kg.graph[source]:
            if (source, target) not in kg_tmp:
                kg_tmp[(source, target)] = set([relation])
            else:
                kg_tmp[(source, target)].update(set([relation]))
    etype = G.new_edge_property('object')
    for (source, target) in kg_tmp:
        e = G.add_edge(source,target)
        etype[e] = kg_tmp[(source, target)]
    return G, etype

def set_args(args):
    args.entity_dim = 100
    args.relation_dim = 100
    args.entity_type_dim = 100
    args.max_path = 3
    args.bandwidth = 3000
    args.bucket_interval = 50
    args.state_history = 2
    args.emb_dropout_rate = 0
    args.disc_hidden = [512, 512]
    args.disc_dropout_rate = 0.3
    args.metadisc_hidden = [512, 256]
    args.metadisc_dropout_rate = 0.3
    args.ac_hidden = [512, 512]
    args.actor_dropout_rate = 0.3
    args.critic_dropout_rate = 0.3
    args.act_dropout = 0.5
    args.target_update = 0.05
    args.gamma = 0.99
    args.factor = 0.9
    args.batch_size = 200

    return args

def get_preferred_curie(curie=None, curie_list=None, name=None, name_list=None):
    if curie or curie_list or name or name_list:
        if curie:
            if isinstance(curie, str):
                try:
                    result = nodesynonymizer.get_canonical_curies(curie)[curie]
                except:
                    result = None
            else:
                print(f"Error: 'curie' should be a string")
                return None
        elif curie_list:
            if isinstance(curie_list, list):
                try:
                    result = nodesynonymizer.get_canonical_curies(curie_list)
                except:
                    result = None
            else:
                print(f"Error: 'curie_list' should be a list")
                return None
        elif name:
            if isinstance(name, str):
                try:
                    result = nodesynonymizer.get_canonical_curies(names=name)[name]
                except:
                    result = None
            else:
                print(f"Error: 'name' should be a string")
                return None    
        else:
            if isinstance(name_list, list):
                try:
                    result = nodesynonymizer.get_canonical_curies(names=name_list)
                except:
                    result = None
            else:
                print(f"Error: 'name_list' should be a list")
                return None         
        return result
    else:
        print(f"Error: No parameter Provided. Please provide given curie(s) or name(s) to one of parameters 'curie', 'curie_list', 'name' or 'name_list'.")
        return None


def id_to_name(curie: str):
    if curie is None:
        return str(None)
    else:
        preferred_result = get_preferred_curie(curie=curie)
        if preferred_result:
            return preferred_result['preferred_name']
        else:
            return str(None)

def name_to_id(name: str):
    if name is None:
        return None
    else:
        preferred_result = get_preferred_curie(name=name)
        if preferred_result:
            return preferred_result['preferred_curie']
        else:
            return str(None)


class DataWrapper(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        n_id, adjs = pickle.load(open(self.paths[idx],'rb'))
        return (n_id, adjs)
