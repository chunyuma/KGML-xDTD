import sys
import os
import graph_tool.all as gt
import argparse
import pickle
import numpy as np
import utils
import pandas as pd
from tqdm import tqdm, trange
from node_synonymizer import NodeSynonymizer
synonymizer = NodeSynonymizer()
import collections
import itertools
import torch
import nltk
nltk.download('omw-1.4')
from pattern.text.en import singularize, pluralize
import yaml
from yaml.loader import SafeLoader

def get_kg2c_node1(node_id, node_name):
    if type(node_id) is str:
        node_id = node_id.replace('UniProt:','UniProtKB:')
    if type(node_id) is str and synonymizer.get_normalizer_results(node_id)[node_id]:
        preferred_node_info = synonymizer.get_normalizer_results(node_id)[node_id]['id']
        return [preferred_node_info['identifier'],preferred_node_info['name'].capitalize(),preferred_node_info['category']]
    if type(node_name) is str and synonymizer.get_normalizer_results(node_name)[node_name]:
        preferred_node_info = synonymizer.get_normalizer_results(node_name)[node_name]['id']
        return [preferred_node_info['identifier'],preferred_node_info['name'].capitalize(),preferred_node_info['category']]  
    else:
        return [None, None, None]


def get_kg2c_node2(node_id, node_name):
    identifier_list = []
    name_list = []
    category_list = []
    if type(node_name) is str:
        node_names = list(set([singularize(node_name),pluralize(node_name)]))
    else:
        node_names = []
    if type(node_id) is str:
        node_id = node_id.replace('UniProt:','UniProtKB:')
    if type(node_id) is str and synonymizer.get_normalizer_results(node_id)[node_id]:
        preferred_node_info = synonymizer.get_normalizer_results(node_id)[node_id]['id']
        identifier_list += [preferred_node_info['identifier']]
        name_list += [preferred_node_info['name'].capitalize()]
        category_list += [preferred_node_info['category']]
    for name in node_names:
        if type(name) is str and synonymizer.get_normalizer_results(name)[name]:
            preferred_node_info = synonymizer.get_normalizer_results(name)[name]['id']
            identifier_list += [preferred_node_info['identifier']]
            name_list += [preferred_node_info['name'].capitalize()]
            category_list += [preferred_node_info['category']]   
    
    identifier_list = list(set(identifier_list))
    name_list = list(set(name_list))
    category_list = list(set(category_list))
    if len(identifier_list) > 0:
        return [identifier_list,name_list,category_list]
    else:
        return [None, None, None]

def extract_and_convert(path_nodes):
    path_nodes = path_nodes[1:-1]
    if len(path_nodes) != 0:
        res = [x for node in path_nodes if get_kg2c_node2(node['id'],node['name'])[0] for x in get_kg2c_node2(node['id'],node['name'])[0]]
        if len(res) == 0:
            return None
        else:
            return res
    else:
        return None

def select_curated_path(all_paths,path_converted_node_index,max_num_path_intermeidate_nodes):
    edge_mat, node_mat = all_paths
    if len(node_mat)==0:
        return None
    else:
        if max_num_path_intermeidate_nodes == 1:
            temp = pd.DataFrame(node_mat.numpy())
            indexes = list(temp.loc[temp[2] == temp[3],:].index)
            edge_mat, node_mat = edge_mat[indexes], node_mat[indexes]
            temp = pd.DataFrame(node_mat.numpy())
            boolean = temp.apply(lambda row: len(set([row[1]]).intersection(set(path_converted_node_index)))>0, axis=1, result_type='expand')
            if len(boolean) != 0:
                indexes = list(temp.loc[boolean,:].index)
            else:
                indexes = []
            if len(indexes) == 0:
                return [[],[]]
            else:
                return [edge_mat[indexes],node_mat[indexes]]
        else:
            temp = pd.DataFrame(node_mat.numpy())
            boolean = temp.apply(lambda row: len(set([row[1],row[2]]).intersection(set(path_converted_node_index)))>1, axis=1, result_type='expand')
            if len(boolean) != 0:
                indexes = list(temp.loc[boolean,:].index)
            else:
                indexes = []
            if len(indexes) == 0:
                return [[],[]]
            else:
                return [edge_mat[indexes],node_mat[indexes]]          
    
class knowledge_graph:

    def __init__(self, data_dir, bandwidth=3000):
        # Load data
        self.bandwidth = bandwidth
        self.entity2id, self.id2entity = utils.load_index(os.path.join(data_dir, 'entity2freq.txt'))
        self.num_entities = len(self.entity2id)
        self.relation2id, self.id2relation = utils.load_index(os.path.join(data_dir, 'relation2freq.txt'))

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
    parser.add_argument("--log_name", type=str, help="log file name", default="step13_2.log")
    parser.add_argument('--drugmechDB_yaml', type=str, help="The full path of 'indication_path.yaml' from DrugMeshDB Github repo")
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default="~/drugmeshdb")
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir,args.log_name))
    logger.info(args)

    ## create output folder
    if not os.path.isdir(args.output_folder):
        os.mkdir(args.output_folder)

    ########################################
    ## read yaml file
    args.logger.info(f"Reading yaml file from {args.drugmechDB_yaml}")
    with open(args.drugmechDB_yaml,'rb') as infile:
        drugmeshdb_data = yaml.load(infile, Loader=SafeLoader)
    drugmeshdb_data = pd.DataFrame([(path['graph']['drug_mesh'], path['graph']['drug'], path['graph']['disease_mesh'], path['graph']['disease'], path['nodes']) for path in drugmeshdb_data])
    drugmeshdb_data.columns = ['drug_mesh', 'drug_name', 'disease_mesh', 'disease_name', 'path_nodes']

    args.logger.info("Start processing drugmeshdb")
    args.logger.info("Step1: mapping drugs and diseases to biological entities used in RTX-KG2c")
    temp = drugmeshdb_data.apply(lambda row: get_kg2c_node1(row[0],row[1])+get_kg2c_node1(row[2],row[3]), axis=1, result_type='expand')
    temp.columns = ['kg2_drug_id','kg2_drug_name','kg2_drug_type','kg2_disease_id','kg2_disease_name','kg2_disease_type']
    drugmeshdb_data = pd.concat([drugmeshdb_data, temp], axis=1)
    bool_select = (~ drugmeshdb_data['kg2_drug_id'].isna()) & (~ drugmeshdb_data['kg2_disease_id'].isna()) & drugmeshdb_data['kg2_drug_type'].isin(['biolink:SmallMolecule','biolink:Drug']) & drugmeshdb_data['kg2_disease_type'].isin(['biolink:Disease','biolink:PhenotypicFeature','biolink:BehavioralFeature','biolink:DiseaseOrPhenotypicFeature'])
    drugmeshdb_data = drugmeshdb_data.loc[bool_select,:].reset_index(drop=True)
    drugmeshdb_data.to_csv(os.path.join(args.output_folder,'drugmeshdb_data_match1.txt'), sep='\t', index=None)

    ########################################
    args.logger.info("Step2: mapping the intermediate nodes to biological entities used in RTX-KG2c")
    kg = knowledge_graph(data_dir, bandwidth=3000)
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
    node_ids = list(kg.entity2id.keys())
    drugmeshdb_data = drugmeshdb_data.loc[(drugmeshdb_data['kg2_drug_id'].isin(node_ids)) & (drugmeshdb_data['kg2_disease_id'].isin(node_ids)),:]
    temp = drugmeshdb_data.apply(lambda row: [extract_and_convert(row[4])], axis=1, result_type='expand')
    temp.columns = ['path_converted_node']
    drugmeshdb_data = pd.concat([drugmeshdb_data, temp], axis=1)
    drugmeshdb_data = drugmeshdb_data.loc[~drugmeshdb_data['path_converted_node'].isna(),:].reset_index(drop=True)
    drugmeshdb_data = drugmeshdb_data.loc[drugmeshdb_data['path_converted_node'].apply(len) > 0,:].reset_index(drop=True)
    temp = drugmeshdb_data.apply(lambda row: [[kg.entity2id.get(x) for x in row[11] if kg.entity2id.get(x)]], axis=1, result_type='expand')
    temp.columns = ['path_converted_node_index']
    drugmeshdb_data = pd.concat([drugmeshdb_data, temp], axis=1)
    drugmeshdb_data.to_csv(os.path.join(args.output_folder,'drugmeshdb_data_match2.txt'), sep='\t', index=None)

    ########################################
    args.logger.info("Step3: merging the duplicate matched drug-disease pairs")
    temp_dict = dict()
    for index in range(len(drugmeshdb_data)):
        source, target = drugmeshdb_data.loc[index,['kg2_drug_id','kg2_disease_id']]
        if (source, target) in temp_dict:
            temp_dict[(source, target)]['drug_mesh'] += [drugmeshdb_data.loc[index,'drug_mesh']]
            temp_dict[(source, target)]['drug_name'] += [drugmeshdb_data.loc[index,'drug_name']]
            temp_dict[(source, target)]['disease_mesh'] += [drugmeshdb_data.loc[index,'disease_mesh']]
            temp_dict[(source, target)]['disease_name'] += [drugmeshdb_data.loc[index,'disease_name']]
            temp_dict[(source, target)]['path_nodes'] += [drugmeshdb_data.loc[index,'path_nodes']]
            temp_dict[(source, target)]['path_converted_node'].update(set(drugmeshdb_data.loc[index,'path_converted_node']))
            temp_dict[(source, target)]['path_converted_node_index'].update(set(drugmeshdb_data.loc[index,'path_converted_node_index']))
        else:
            temp_dict[(source, target)] = dict()
            temp_dict[(source, target)]['kg2_drug_name'] =  drugmeshdb_data.loc[index,'kg2_drug_name']
            temp_dict[(source, target)]['kg2_drug_type'] =  drugmeshdb_data.loc[index,'kg2_drug_type']
            temp_dict[(source, target)]['kg2_disease_name'] =  drugmeshdb_data.loc[index,'kg2_disease_name']
            temp_dict[(source, target)]['kg2_disease_type'] =  drugmeshdb_data.loc[index,'kg2_disease_type']
            temp_dict[(source, target)]['drug_mesh'] = [drugmeshdb_data.loc[index,'drug_mesh']]
            temp_dict[(source, target)]['drug_name'] = [drugmeshdb_data.loc[index,'drug_name']]
            temp_dict[(source, target)]['disease_mesh'] = [drugmeshdb_data.loc[index,'disease_mesh']]
            temp_dict[(source, target)]['disease_name'] = [drugmeshdb_data.loc[index,'disease_name']]
            temp_dict[(source, target)]['path_nodes'] = [drugmeshdb_data.loc[index,'path_nodes']]
            temp_dict[(source, target)]['path_converted_node'] = set(drugmeshdb_data.loc[index,'path_converted_node'])
            temp_dict[(source, target)]['path_converted_node_index'] = set(drugmeshdb_data.loc[index,'path_converted_node_index'])

    temp = [(temp_dict[(source, target)]['drug_mesh'],temp_dict[(source, target)]['drug_name'],temp_dict[(source, target)]['disease_mesh'],temp_dict[(source, target)]['disease_name'],temp_dict[(source, target)]['path_nodes'],source,temp_dict[(source, target)]['kg2_drug_name'],temp_dict[(source, target)]['kg2_drug_type'],target,temp_dict[(source, target)]['kg2_disease_name'],temp_dict[(source, target)]['kg2_disease_type'],list(temp_dict[(source, target)]['path_converted_node']),list(temp_dict[(source, target)]['path_converted_node_index'])) for (source, target) in temp_dict]
    drugmeshdb_data = pd.DataFrame(temp)
    drugmeshdb_data.columns = ['drug_mesh',
                               'drug_name',
                               'disease_mesh',
                               'disease_name',
                               'path_nodes',
                               'kg2_drug_id',
                               'kg2_drug_name',
                               'kg2_drug_type',
                               'kg2_disease_id',
                               'kg2_disease_name',
                               'kg2_disease_type',
                               'path_converted_node',
                               'path_converted_node_index']
    drugmeshdb_data.to_csv(os.path.join(args.output_folder,'drugmeshdb_data_match3.txt'), sep='\t', index=None)
    
    ########################################
    args.logger.info("Extracting all KG-based paths between the matched DrugmeshDB drug-disease pairs")
    res_all_paths = dict()
    filter_edges = [kg.relation2id[edge] for edge in ['biolink:related_to','biolink:biolink:part_of','biolink:coexists_with','biolink:contraindicated_for'] if G.relation2id.get(edge)]
    for index1 in range(len(drugmeshdb_data)):
        print(index1, flush=True)
        source, target = drugmeshdb_data.loc[index1,['kg2_drug_id','kg2_disease_id']]
        all_paths = [list(path) for path in gt.all_paths(G, check_curie(source)[1], check_curie(target)[1], cutoff=3)]
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
        edge_mat = torch.tensor(relation_paths)
        node_mat = torch.tensor(np.array(entity_paths).astype(int))
        temp = pd.DataFrame(edge_mat.numpy())
        keep_index = list(temp.loc[~(temp[1].isin(filter_edges) | temp[2].isin(filter_edges) | temp[3].isin(filter_edges)),:].index)
        res_all_paths[(source,target)] = [edge_mat[keep_index],node_mat[keep_index]]
    with open(os.path.join(args.output_folder,'res_all_paths.pkl'),'wb') as outfile:
        pickle.dump(res_all_paths, outfile)

    ########################################
    args.logger.info("Matching Drugmesh paths to 3-hop KG-based paths") 
    test = dict()
    for index in range(len(drugmeshdb_data)):
        path_converted_node_index = drugmeshdb_data.loc[index,'path_converted_node_index']
        max_num_path_intermeidate_nodes = max([len(x[1:-1]) for x in drugmeshdb_data.loc[index,'path_nodes']])
        kg2_drug_id = drugmeshdb_data.loc[index,'kg2_drug_id']
        kg2_disease_id = drugmeshdb_data.loc[index,'kg2_disease_id']
        all_paths = res_all_paths[(kg2_drug_id,kg2_disease_id)]
        res = select_curated_path(all_paths,path_converted_node_index,max_num_path_intermeidate_nodes)
        if res and len(res[1]) != 0:
            test[(kg2_drug_id,kg2_disease_id)] = res
    with open(os.path.join(args.output_folder,'match_paths.pkl'),'wb') as outfile:
        pickle.dump(test,outfile)

    temp_index_list = []
    for index in range(len(drugmeshdb_data)):
        kg2_drug_id = drugmeshdb_data.loc[index,'kg2_drug_id']
        kg2_disease_id = drugmeshdb_data.loc[index,'kg2_disease_id']
        if (kg2_drug_id,kg2_disease_id) in test:
            temp_index_list += [index]
    drugmeshdb_data_match_paths = drugmeshdb_data.loc[temp_index_list,:].reset_index(drop=True)
    drugmeshdb_data_match_paths.to_csv(os.path.join(args.output_folder,'drugmeshdb_data_match_paths.txt'), sep='\t', index=None)