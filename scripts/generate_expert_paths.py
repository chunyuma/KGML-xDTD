import graph_tool.all as gt
import os, sys
import argparse
import utils
import pickle
import collections
import itertools
import torch
import pandas as pd
import sqlite3
from tqdm import tqdm
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

def load_data_from_ngd_database(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM curie_to_pmids", conn)
    return df

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

def check_path(path):
    if len(path) == 2:
        return (path, None)
    else:
        if path[1] in ngd_mapping_dict and path[2] in ngd_mapping_dict:
            return (path, utils.calculate_ngd([ngd_mapping_dict[path[1]],ngd_mapping_dict[path[2]]]))
        else:
            return None

def extract_all_paths_between_two_nodes_in_parallel(this):
    source, target, source_id, target_id = this
    if source_id and target_id:
        return [(source, target),[(check_path(list(path))[0], check_path(list(path))[1]) for path in gt.all_paths(G, source_id, target_id, cutoff=args.max_path-1) if check_path(list(path))]]
    else:
        return [(source, target),[]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default="../log_folder")
    parser.add_argument("--log_name", type=str, help="log file name", default="step3_5.log")
    parser.add_argument('--data_dir', type=str, help='Full path of data folder', default='../data')
    parser.add_argument('--expert_dir_name', type=str, help='The name of expert path directory', default='expert_path_files')
    parser.add_argument('--ngd_db_path', type=str, help='The path of NGD database', default='../bkg_rtxkg2c_v2.7.3/relevant_dbs/curie_to_pmids_v1.0_KG2.7.3.sqlite')
    parser.add_argument('--ngd_threshold', type=float, help='NGD threshold to filter paths for expert paths', default=0.6)
    parser.add_argument('--batch_size', type=int, help='Number of batch size for parallel running', default=500)
    parser.add_argument('--process', type=float, help='Number of processes used for parallel running', default=100)
    parser.add_argument('--bandwidth', type=int, help='Maximum number of neighbors', default=3000)
    parser.add_argument('--max_path', type=int, help='Maximum length of path', default=3)
    args = parser.parse_args()

    args.path_dir = os.path.join(args.data_dir,args.expert_dir_name)

    logger = utils.get_logger(os.path.join(args.log_dir,args.log_name))
    logger.info(args)

    ## read pruned knowledge graph with specified neighbor bandwidth
    kg = knowledge_graph(args.data_dir, bandwidth=args.bandwidth)

    ## get predicate depth in biolink model relation hierarchy 
    predicate_list = list(kg.relation2id.keys())
    predicate_list.remove('DUMMY_RELATION')
    predicate_list.remove('SELF_LOOP_RELATION')
    predicate_list.remove('biolink:has_effect')
    predicate_list.remove('biolink:has_no_effect')
    predicate_depth = utils.get_depth_of_predicate(predicate_list, biolink_version='2.1.0')

    ## load data from NGD database
    if not os.path.exists(os.path.join(args.path_dir, "ngd_mapping.pkl")):
        if os.path.exists(args.ngd_db_path):
            ngd_df = load_data_from_ngd_database(args.ngd_db_path)
        else:
            logger.error(f"No file found from {args.ngd_db_path}")
            exit()
        ngd_mapping_dict = {kg.entity2id[ngd_df.loc[index,'curie']]:eval(ngd_df.loc[index,'pmids']) for index in range(len(ngd_df)) if ngd_df.loc[index,'curie'] in kg.entity2id}
        with open(os.path.join(args.path_dir, "ngd_mapping.pkl"), 'wb') as outfile:
            pickle.dump(ngd_mapping_dict, outfile)
    else:
        with open(os.path.join(args.path_dir, "ngd_mapping.pkl"), 'rb') as infile:
            ngd_mapping_dict = pickle.load(infile)

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

    ## read checked reachable expert path files
    checked_expert_paths = pd.read_csv(os.path.join(args.path_dir, f"reachable_expert_paths_max{args.max_path}.txt"), sep='\t', header=0)
    reachable_expert_paths = checked_expert_paths.loc[checked_expert_paths['reachable'],['source','n2','target']].reset_index(drop=True)
    ## filter drug-target by NGD
    temp = reachable_expert_paths[['source','n2']].drop_duplicates().reset_index(drop=True)
    ngd_list = [utils.calculate_ngd([ngd_mapping_dict.get(pair[0]),ngd_mapping_dict.get(pair[1])]) for pair in temp.apply(lambda row: [kg.entity2id[row[0]],kg.entity2id[row[1]]], axis=1, result_type='expand').to_numpy()]
    temp = pd.concat([temp,pd.DataFrame(ngd_list)], axis=1).dropna().reset_index(drop=True)
    temp.columns = ['source', 'n2', 'ngd_score']
    temp = temp.loc[temp['ngd_score'] <= args.ngd_threshold,:].reset_index(drop=True)
    reachable_expert_paths = pd.merge(reachable_expert_paths,temp,on=['source','n2']).reset_index(drop=True)
    reachable_expert_paths = reachable_expert_paths.drop(['ngd_score'],axis=1)
    ################################
    temp_dict = dict()
    temp_df = reachable_expert_paths[['n2','target']].drop_duplicates().reset_index(drop=True)
    temp_df_list = list(temp_df.apply(lambda row: [row[0],row[1], check_curie(row[0])[1], check_curie(row[1])[1]], axis=1))
    batch =list(range(0,len(temp_df_list),args.batch_size))
    batch.append(len(temp_df_list))
    logger.info(f'Total batch for extracting paths: {len(batch)-1}')
    ## run each batch in parallel
    for i in tqdm(range(len(batch))):
        if((i+1)<len(batch)):
            logger.info(f'Here is batch{i+1}')
            start = batch[i]
            end = batch[i+1]
            if args.process == -1:
                with multiprocessing.Pool() as executor:
                    temp_dict.update({x[0]:x[1] for x in executor.map(extract_all_paths_between_two_nodes_in_parallel,temp_df_list[start:end])})
            else:
                with multiprocessing.Pool(processes=args.process) as executor:
                    temp_dict.update({x[0]:x[1] for x in executor.map(extract_all_paths_between_two_nodes_in_parallel,temp_df_list[start:end])})
            with open(os.path.join(args.path_dir, f"temp_dict_backup.pkl"), 'wb') as outfile:
                pickle.dump(temp_dict, outfile)
    with open(os.path.join(args.path_dir, f"temp_dict_backup.pkl"), 'wb') as outfile:
        pickle.dump(temp_dict, outfile)

    expert_demonstration_paths = dict()
    for index in tqdm(range(len(reachable_expert_paths))):
        source, e2, target = reachable_expert_paths.loc[index,:]
        if (e2, target) in temp_dict:
            temp = [([check_curie(source)[1]]+ x[0], x[1]) for x in temp_dict[(e2, target)]]
            if len(temp) > 0:
                if (source,target) in expert_demonstration_paths:
                    expert_demonstration_paths[(source,target)] += temp
                else:
                    expert_demonstration_paths[(source,target)] = temp
    with open(os.path.join(args.path_dir, f"expert_demonstration_paths_max{args.max_path}_raw.pkl"), 'wb') as outfile:
        pickle.dump(expert_demonstration_paths, outfile)
    logger.info(f"{sum([len(expert_demonstration_paths[key]) for key in expert_demonstration_paths])} expert demonstration paths with max length {args.max_path} have been found from {len(expert_demonstration_paths)} true positive drug-disease pairs")
    
    temp_del_list = []
    for (source, target) in expert_demonstration_paths:
        temp = [x for x in expert_demonstration_paths[(source, target)] if (x[1] and x[1] <= args.ngd_threshold)]
        if len(temp) > 0:
            expert_demonstration_paths[(source, target)] = temp
        else:
            temp_del_list += [(source, target)]
    for (source, target) in temp_del_list:
        del expert_demonstration_paths[(source, target)]
    logger.info(f"After filtering the paths with NGD scores <= {args.ngd_threshold}, {sum([len(expert_demonstration_paths[key]) for key in expert_demonstration_paths])} expert demonstration paths with max length {args.max_path} have been found from {len(expert_demonstration_paths)} true positive drug-disease pairs")

    ## add relation into expert demonstration paths
    for (source, target) in expert_demonstration_paths:
        all_paths = expert_demonstration_paths[(source, target)]
        all_paths_temp = []
        for item in all_paths:
            path_temp = []
            path, ngd_score = item
            for index in range(len(path)-1):
                if index == 0:
                    path_temp += [path[index], list(etype[G.edge(path[index], path[index+1])]), path[index+1]]
                else:
                    path_temp += [list(etype[G.edge(path[index], path[index+1])]), path[index+1]]
            all_paths_temp += [(path_temp, ngd_score)]
        expert_demonstration_paths[(source, target)] = all_paths_temp

    expert_demonstration_paths_translate = dict()
    for (source, target) in expert_demonstration_paths:
        all_paths = expert_demonstration_paths[(source, target)]
        all_paths_temp = []
        keep_indexes = []
        for keep_idx, item in enumerate(all_paths):
            path_temp = []
            path, ngd_score = item
            for index in range(len(path)):
                if type(path[index]) is list:
                    ## remove 'biolink:coexists_with', 'biolink:related_to', 'biolink:part_of' which might not have signficant contribution to DTD explanation
                    relations = [kg.id2relation[relation_id] for relation_id in path[index] if kg.id2relation[relation_id]!='biolink:coexists_with' and  kg.id2relation[relation_id]!='biolink:part_of' and  kg.id2relation[relation_id]!='biolink:related_to']
                    if len(relations) > 0:
                        max_value_temp = max([predicate_depth[relation] for relation in relations])
                        relations = [relation for relation in relations if predicate_depth[relation]==max_value_temp]
                        path_temp += [relations]
                    else:
                        path_temp = []
                        break
                else:
                    path_temp += [kg.id2entity[path[index]]]
            if len(path_temp) != 0:
                keep_indexes += [keep_idx]
                all_paths_temp += [(path_temp, ngd_score)]
        if len(all_paths_temp) != 0:
            expert_demonstration_paths[(source, target)] = [expert_demonstration_paths[(source, target)][x] for x in keep_indexes]
            expert_demonstration_paths_translate[(source, target)] = all_paths_temp

    out_pairs = expert_demonstration_paths.keys() - expert_demonstration_paths_translate.keys()
    for pair in out_pairs:
        del expert_demonstration_paths[pair]

    logger.info(f"After removing 'biolink:coexists_with', 'biolink:related_to', 'biolink:part_of' in expert demonstration paths and only keeping the predicates with maximum depths, \
    {len(expert_demonstration_paths_translate)} true positive drug-disease pairs have at least one expert demonstration paths with max length {args.max_path}")
    logger.info(f"{sum([len(expert_demonstration_paths_translate[key]) for key in expert_demonstration_paths_translate])} expert demonstration paths (considers nodes only) with max length {args.max_path} have been found from {len(expert_demonstration_paths_translate)} true positive drug-disease pairs")

    with open(os.path.join(args.path_dir, f"expert_demonstration_paths_max{args.max_path}_filtered.pkl"), 'wb') as outfile:
        pickle.dump(expert_demonstration_paths, outfile)

    with open(os.path.join(args.path_dir, f"expert_demonstration_paths_translate_max{args.max_path}_filtered.pkl"), 'wb') as outfile:
        pickle.dump(expert_demonstration_paths_translate, outfile)

    entity_expert_paths = []
    relation_expert_paths = []
    for (source, target) in expert_demonstration_paths_translate:
        all_paths = expert_demonstration_paths_translate[(source, target)]
        for item in all_paths:
            path, ngd_score = item
            flattened_paths = list(itertools.product(*map(lambda x: [x] if type(x) is str else x, path)))
            for flattened_path in flattened_paths:
                if len(flattened_path) == 7:
                    relation_expert_paths += [[kg.relation2id['SELF_LOOP_RELATION']] + [kg.relation2id[x] for index, x in enumerate(flattened_path) if index%2==1]]
                    entity_expert_paths += [[kg.entity2id[x] for index, x in enumerate(flattened_path) if index%2==0]]
                elif len(flattened_path) == 5:
                    relation_expert_paths += [[kg.relation2id['SELF_LOOP_RELATION']] + [kg.relation2id[x] for index, x in enumerate(flattened_path) if index%2==1] + [kg.relation2id['SELF_LOOP_RELATION']]]
                    entity_expert_paths += [[kg.entity2id[x] for index, x in enumerate(flattened_path) if index%2==0] + [kg.entity2id[flattened_path[-1]]]]
                else:
                    logger.info(f"Found weird path: {flattened_path}")
    expert_paths = [torch.tensor(relation_expert_paths),torch.tensor(entity_expert_paths)]

    logger.info(f"{expert_paths[1].shape[0]} expert demonstration paths (considers nodes and edges) with max length {args.max_path} have been found from {len(expert_demonstration_paths_translate)} true positive drug-disease pairs")

    with open(os.path.join(args.path_dir, f"expert_demonstration_relation_entity_max{args.max_path}_filtered.pkl"), 'wb') as outfile:
        pickle.dump(expert_paths, outfile)