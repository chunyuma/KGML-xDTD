import os, sys
import graph_tool.all as gt
import argparse
import utils
import pickle
import collections
import pandas as pd
import sqlite3
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

def convert_to_canonicalized_curie(curie):
    if not synonymizer.get_canonical_curies(curie)[curie]:
        return None
    else:
        return synonymizer.get_canonical_curies(curie)[curie]['preferred_curie']

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

def check_direct_connect(pair):
    if check_curie(pair[0])[1] is not None and check_curie(pair[1])[1] is not None:
        return len([path for path in gt.all_paths(G, check_curie(pair[0])[1], check_curie(pair[1])[1], cutoff=1)]) > 0
    else:
        return False

def check_connect(params):
    source, target, cutoff = params
    print([source, target], flush=True)
    if check_curie(source)[1] is not None and check_curie(target)[1] is not None:
        for path in gt.all_paths(G, check_curie(source)[1], check_curie(target)[1], cutoff=cutoff):
            return True
        return False
    else:
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default="../log_folder")
    parser.add_argument("--log_name", type=str, help="Log file name", default="step3_4.log")
    parser.add_argument('--data_dir', type=str, help='Full path of data folder', default='../data')
    parser.add_argument('--expert_dir_name', type=str, help='The name of expert path directory', default='expert_path_files')
    parser.add_argument('--bandwidth', type=int, help='Maximum number of neighbors', default=3000)
    parser.add_argument('--max_path', type=int, help='Maximum length of path', default=3)
    args = parser.parse_args()
    
    args.path_dir = os.path.join(args.data_dir,args.expert_dir_name)

    logger = utils.get_logger(os.path.join(args.log_dir,args.log_name))
    logger.info(args)

    ## read pruned knowledge graph with specified neighbor bandwidth
    kg = knowledge_graph(args.data_dir, bandwidth=args.bandwidth)
    
    ## store knowledge graph in graph tool
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
    # G.save(os.path.join(args.path_dir, f'pruned_kg_bandwidth{args.bandwidth}.xml.gz'))

    ## read potential expert paths
    args.expert_path_file = os.path.join(args.path_dir,'p_expert_paths_combined.txt')
    p_expert_paths = pd.read_csv(args.expert_path_file, sep='\t', header=0)
    for col in p_expert_paths.columns:
        p_expert_paths.loc[p_expert_paths[col].isna(),col] = None
    p_expert_paths = p_expert_paths.apply(lambda row: [convert_to_canonicalized_curie(row[0]),convert_to_canonicalized_curie(row[1])], axis=1, result_type='expand').drop_duplicates().reset_index(drop=True)


    ## check if there is an edge directly connecting between drug entities and protein entities in the pruned knolwedge graph 
    p_expert_paths = p_expert_paths.apply(lambda row: [row[0],row[1], check_direct_connect([row[0],row[1]])], axis=1, result_type='expand')
    p_expert_paths = p_expert_paths.loc[p_expert_paths[2],[0,1]].reset_index(drop=True)
    p_expert_paths = p_expert_paths.rename(columns={0:'source',1:'n2'})
    
    ## merge target disease into the potential expert paths
    tp_pairs = pd.read_csv(os.path.join(args.data_dir,'tp_pairs.txt'), sep='\t', header=0)
    p_expert_paths_with_disease = p_expert_paths.merge(tp_pairs, left_on='source', right_on='source')
    p_expert_paths_with_disease = p_expert_paths_with_disease.reset_index(drop=True)

    temp = p_expert_paths_with_disease[['n2','target']].drop_duplicates()
    logger.info(f"{len(temp)} n2-target pairs need to be checked for path with length {args.max_path - 1}")
    temp = temp.reset_index(drop=True)
    temp['cutoff'] = args.max_path - 1
    iters = list(temp.to_records(index=False))
    reachable = [elem for elem in map(check_connect, iters)]
    temp['reachable'] = reachable
    temp = pd.merge(p_expert_paths_with_disease,temp,how='left', left_on=['n2','target'], right_on=['n2','target'])
    temp = temp.drop(['cutoff'], axis=1)
    temp.loc[temp['reachable'].isna(),'reachable'] = False
    p_expert_paths_with_disease = temp
    p_expert_paths_with_disease.to_csv(os.path.join(args.path_dir, f"reachable_expert_paths_max{args.max_path}.txt"),sep='\t',index=None)

    ## extract reachable drug-disease pairs form experet demonstration paths
    reachable_tp_pairs = p_expert_paths_with_disease.loc[p_expert_paths_with_disease['reachable'],['source','target']].drop_duplicates().reset_index(drop=True)
    temp = {(reachable_tp_pairs.loc[index,'source'],reachable_tp_pairs.loc[index,'target']):1 for index in range(len(reachable_tp_pairs))}
    check_pairs = [(tp_pairs.loc[index,'source'],tp_pairs.loc[index,'target']) for index in range(len(tp_pairs)) if (tp_pairs.loc[index,'source'],tp_pairs.loc[index,'target']) not in temp]
    temp = pd.DataFrame(check_pairs)
    logger.info(f"{len(temp)} source-target pairs need to be checked for path with length {args.max_path}")
    temp['cutoff'] = args.max_path
    iters = list(temp.to_records(index=False))
    reachable = [elem for elem in map(check_connect, iters)]
    temp['reachable'] = reachable
    temp = temp.drop(['cutoff'], axis=1)
    reachable_tp_pairs = pd.concat([reachable_tp_pairs,temp.loc[temp['reachable'],[0,1]].rename(columns={0:'source',1:'target'})]).reset_index(drop=True)
    unreachable_tp_pairs = temp.loc[~temp['reachable'],[0,1]].rename(columns={0:'source',1:'target'}).reset_index(drop=True)

    ## save reachable_tp_pairs and unreachable_tp_pairs to text files
    reachable_tp_pairs.to_csv(os.path.join(args.path_dir, f"reachable_tp_pairs_max{args.max_path}.txt"), sep='\t',index=None)
    unreachable_tp_pairs.to_csv(os.path.join(args.path_dir, f"unreachable_tp_pairs_max{args.max_path}.txt"), sep='\t',index=None)