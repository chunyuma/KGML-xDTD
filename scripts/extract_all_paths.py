import os, sys
import graph_tool.all as gt
import argparse
import utils
import pickle
import collections
import itertools
import torch
import numpy as np
import pandas as pd
import math
import multiprocessing
from node_synonymizer import NodeSynonymizer
synonymizer = NodeSynonymizer()

class knowledge_graph:

    def __init__(self, data_dir, bandwidth=3000):
        # Load data
        self.bandwidth = bandwidth
        self.entity2id, self.id2entity = utils.load_index(os.path.join(data_dir, 'entity2freq.txt'))
        logger.info('Total {} entities loaded'.format(len(self.entity2id)))
        self.num_entities = len(self.entity2id)
        self.relation2id, self.id2relation = utils.load_index(os.path.join(data_dir, 'relation2freq.txt'))
        logger.info('Total {} relations loaded'.format(len(self.relation2id)))

        all_nodes = set()
        with open(os.path.join(data_dir, 'graph_edges.txt'), 'rb') as f:
            header_row = f.readline().decode()
            for line in f:
                all_nodes.update([line.decode().strip().split()[0]])
                all_nodes.update([line.decode().strip().split()[1]])
        with open(os.path.join(data_dir, 'all_graph_nodes_info.txt'), 'rb') as f:
            header_row = f.readline().decode()
            curie_to_type = {line.decode().strip().split()[0]:line.decode().strip().split()[1] for line in f}

        # Load graph structures
        adj_list_path = os.path.join(data_dir, 'adj_list.pkl')
        with open(adj_list_path, 'rb') as f:
            self.adj_list = pickle.load(f)

        self.page_rank_scores = self.load_page_rank_scores(os.path.join(data_dir, 'kg.pgrk'))

        self.graph = {source:self.get_action_space(source) for source in range(self.num_entities)}

    def load_page_rank_scores(self, input_path):
        pgrk_scores = collections.defaultdict(float)
        with open(input_path) as f:
            for line in f:
                entity, score = line.strip().split('\t')
                entity_id = self.entity2id[entity.strip()]
                score = float(score)
                pgrk_scores[entity_id] = score
        return pgrk_scores


    def get_action_space(self, source):
        action_space = []
        if source in self.adj_list:
            for relation in self.adj_list[source]:
                targets = self.adj_list[source][relation]
                for target in targets:
                    action_space.append((relation, target))
            if len(action_space) + 1 >= self.bandwidth:
                # Base graph pruning
                sorted_action_space = sorted(action_space, key=lambda x: self.page_rank_scores[x[1]], reverse=True)
                action_space = sorted_action_space[:self.bandwidth]
        return action_space

def check_curie(curie):
    if curie is None:
        return (None, None)
    res = synonymizer.get_canonical_curies(curie)[curie]
    if res is not None:
        preferred_curie = synonymizer.get_canonical_curies(curie)[curie]['preferred_curie']
    else:
        preferred_curie = None
    if preferred_curie in kg.entity2id:
        return (preferred_curie, kg.entity2id[preferred_curie])
    else:
        return (preferred_curie, None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## folder parameters
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default="../log_folder")
    parser.add_argument("--log_name", type=str, help="log file name", default="step13.log")
    parser.add_argument('--data_dir', type=str, help='Full path of data folder', default='../data')

    ## knowledge graph and environment parameters
    parser.add_argument('--max_path', type=int, help='Maximum length of path', default=3)
    parser.add_argument('--bandwidth', type=int, help='Maximum number of neighbors', default=3000)

    # additional parameters
    parser.add_argument('--every_save', type=int, help='How often to save results (default: 400)', default=400)

    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir,args.log_name))
    args.logger = logger
    logger.info(args)

    kg = knowledge_graph(args.data_dir, bandwidth=args.bandwidth)
    val_pairs = pd.read_csv(os.path.join(args.data_dir,'RL_model_train_val_test_data','val_pairs.txt'),'\t',header=0)
    test_pairs = pd.read_csv(os.path.join(args.data_dir,'RL_model_train_val_test_data','test_pairs.txt'),'\t',header=0)
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
    G.edge_properties['edge_type'] = etype


    if not os.path.isdir(os.path.join(args.data_dir,'expert_path_files','all_paths_of_drug_disease_pairs')):
        os.mkdir(os.path.join(args.data_dir,'expert_path_files','all_paths_of_drug_disease_pairs'))

    logger.info(f'extracting all paths of all drug-disease pairs in val_set')
    if not os.path.isdir(os.path.join(args.data_dir,'expert_path_files','all_paths_of_drug_disease_pairs','val_set')):
        os.mkdir(os.path.join(args.data_dir,'expert_path_files','all_paths_of_drug_disease_pairs','val_set'))
    val_all_paths = dict()
    for index1 in range(len(val_pairs)):
        logger.info(f'pair{index1+1}')
        if index1 % args.every_save == 0 and index1!=0:
            batch_num = index1/args.every_save
            with open(os.path.join(args.data_dir,'expert_path_files','all_paths_of_drug_disease_pairs','val_set', f"batch{int(batch_num)}.pkl"),'wb') as outfile:
                pickle.dump(val_all_paths,outfile)
            val_all_paths = dict()
        source, target = val_pairs.loc[index1,['source','target']]
        all_paths = [list(path) for path in gt.all_paths(G, check_curie(source)[1], check_curie(target)[1], cutoff=args.max_path)]
        entity_paths = []
        relation_paths = []
        for path in all_paths:
            path_temp = []
            for index2 in range(len(path)-1):
                if index2 == 0:
                    path_temp += [path[index2], list(etype[G.edge(path[index2], path[index2+1])]), path[index2+1]]
                else:
                    path_temp += [list(etype[G.edge(path[index2], path[index2+1])]), path[index2+1]]
            flattened_paths = list(itertools.product(*map(lambda x: [x] if type(x) is not list else x, path_temp)))
            for flattened_path in flattened_paths:
                if len(flattened_path) == 7:
                    relation_paths += [[kg.relation2id['SELF_LOOP_RELATION']] + [x for index3, x in enumerate(flattened_path) if index3%2==1]]
                    entity_paths += [[x for index3, x in enumerate(flattened_path) if index3%2==0]]
                elif len(flattened_path) == 5:
                    relation_paths += [[kg.relation2id['SELF_LOOP_RELATION']] + [x for index3, x in enumerate(flattened_path) if index3%2==1] + [kg.relation2id['SELF_LOOP_RELATION']]]
                    entity_paths += [[x for index3, x in enumerate(flattened_path) if index3%2==0] + [flattened_path[-1]]]
                else:
                    logger.info(f"Found weird path: {flattened_path}")
        val_all_paths[(source,target)] = [torch.tensor(relation_paths),torch.tensor(np.array(entity_paths).astype(int))]
    batch_num = math.ceil(index1/args.every_save)
    with open(os.path.join(args.data_dir,'expert_path_files','all_paths_of_drug_disease_pairs','val_set', f"batch{int(batch_num)}.pkl"),'wb') as outfile:
        pickle.dump(val_all_paths,outfile)

    logger.info(f'extracting all paths of all drug-disease pairs in test_set')
    if not os.path.isdir(os.path.join(args.data_dir,'expert_path_files','all_paths_of_drug_disease_pairs','test_set')):
        os.mkdir(os.path.join(args.data_dir,'expert_path_files','all_paths_of_drug_disease_pairs','test_set'))
    test_all_paths = dict()
    for index1 in range(len(test_pairs)):
        logger.info(f'pair{index1+1}')
        if index1 % args.every_save == 0 and index1!=0:
            batch_num = index1/args.every_save
            with open(os.path.join(args.data_dir,'expert_path_files','all_paths_of_drug_disease_pairs','test_set', f"batch{int(batch_num)}.pkl"),'wb') as outfile:
                pickle.dump(test_all_paths,outfile)
            test_all_paths = dict()
        source, target = test_pairs.loc[index1,['source','target']]
        all_paths = [list(path) for path in gt.all_paths(G, check_curie(source)[1], check_curie(target)[1], cutoff=args.max_path)]
        entity_paths = []
        relation_paths = []
        for path in all_paths:
            path_temp = []
            for index2 in range(len(path)-1):
                if index2 == 0:
                    path_temp += [path[index2], list(etype[G.edge(path[index2], path[index2+1])]), path[index2+1]]
                else:
                    path_temp += [list(etype[G.edge(path[index2], path[index2+1])]), path[index2+1]]
            flattened_paths = list(itertools.product(*map(lambda x: [x] if type(x) is not list else x, path_temp)))
            for flattened_path in flattened_paths:
                if len(flattened_path) == 7:
                    relation_paths += [[kg.relation2id['SELF_LOOP_RELATION']] + [x for index3, x in enumerate(flattened_path) if index3%2==1]]
                    entity_paths += [[x for index3, x in enumerate(flattened_path) if index3%2==0]]
                elif len(flattened_path) == 5:
                    relation_paths += [[kg.relation2id['SELF_LOOP_RELATION']] + [x for index3, x in enumerate(flattened_path) if index3%2==1] + [kg.relation2id['SELF_LOOP_RELATION']]]
                    entity_paths += [[x for index3, x in enumerate(flattened_path) if index3%2==0] + [flattened_path[-1]]]
                else:
                    logger.info(f"Found weird path: {flattened_path}")
        test_all_paths[(source,target)] = [torch.tensor(relation_paths),torch.tensor(np.array(entity_paths).astype(int))]
    batch_num = math.ceil(index1/args.every_save)
    with open(os.path.join(args.data_dir,'expert_path_files','all_paths_of_drug_disease_pairs','test_set', f"batch{int(batch_num)}.pkl"),'wb') as outfile:
        pickle.dump(test_all_paths,outfile)





