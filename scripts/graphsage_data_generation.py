## This script is used to generate required input files (eg. G.json, id_map.json, class_map.json, please see https://github.com/williamleif/GraphSAGE
# for more details) as well as the mapping files for running Graphsage
from __future__ import print_function

import json
import numpy as np
import pandas as pd
import random
import os
import sys
import argparse
import multiprocessing
from itertools import chain
import pickle
import utils

## setting functions compatible with parallel running
def initialize_node(this):

    node, has_emb_file = this
    if not has_emb_file:
        return [{'test': False, 'id': int(node-1), 'feature': [], 'label': nodeclass_data_index[id2entity[node]][1], 'val': (int(node-1) in valid)}]
    else:
        return [{'test': False, 'id': int(node-1), 'feature': [], 'label': nodeclass_data_index[id2entity[node]][1], 'val': (int(node-1) in valid)}]

def initialize_edge(index):

    return [{'test_removed': False, 'train_removed': False, 'source': int(graph_data.loc[index,'id1']), 'target': int(graph_data.loc[index,'id2'])}]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default="../log_dir")
    parser.add_argument("--log_name", type=str, help="log file name", default="step6_1.log")
    parser.add_argument('--data_dir', type=str, help='Full path of data folder.', default='../data')
    parser.add_argument("--seed", type=int, help="Random seed (default: 1023)", default=1023)
    parser.add_argument("--emb_file", type=str, help="The full path of initial embedding file", default=None)
    parser.add_argument("--feature_dim", type=int, help="The node feature dimension", default=256)
    parser.add_argument("--process", type=int, help="Number of processes to be used", default=-1)
    parser.add_argument("--validation_percent", type=float, help="The percentage of validation data (default: 0.3)", default=0.3)
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default="../data/graphsage_input")

    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir,args.log_name))
    logger.info(args)

    #create output directory
    if not os.path.isdir(args.output_folder):
        os.mkdir(args.output_folder)

    #read the graph file
    with open(os.path.join(args.data_dir, 'graph_edges.txt'),'r') as f:
        graph_data = pd.read_csv(f,sep='\t')
        graph_data = graph_data[['source','target']].drop_duplicates().reset_index(drop=True)

    # read the text embedding file
    if args.emb_file is not None:
        with open(args.emb_file,'rb') as infile:
            emb_file = pickle.load(infile)
    else:
        emb_file = None

    # generate node label vector
    all_graph_nodes_info = pd.read_csv(os.path.join(args.data_dir, 'all_graph_nodes_info.txt'), sep='\t', header=0)
    id_to_type = {all_graph_nodes_info.loc[index, 'id']:all_graph_nodes_info.loc[index, 'category'] for index in range(len(all_graph_nodes_info))}
    type2id, id2type = utils.load_index(os.path.join(args.data_dir, 'type2freq.txt'))
    entity2id, id2entity = utils.load_index(os.path.join(args.data_dir, 'entity2freq.txt'))
    del entity2id['DUMMY_ENTITY']
    # for entity in entity2id:
    #     if entity not in id_to_type:
    #         id_to_type[entity] = entity
    #         type2id[entity] = len(type2id)

    nodeclass_data_index = id_to_type
    for curie, node_cls in nodeclass_data_index.items():
        temp = np.zeros(len(type2id)).tolist()
        temp[type2id[node_cls]] = temp[type2id[node_cls]] + 1
        nodeclass_data_index[curie] = [node_cls, temp]

    ## index - 1 is to make sure there is no mismatch when we include the external intitial embedding
    id_map = pd.DataFrame([(curie, index-1) for curie, index in entity2id.items()]).rename(columns={0: 'curie', 1: 'id'})

    #output the id map file
    id_map.to_csv(os.path.join(args.output_folder, 'id_map.txt'),sep='\t',index=None)

    #output the category mapping file
    temp = dict()
    for key, value in nodeclass_data_index.items():
            if value[0] not in temp:
                temp[value[0]] = value[1]

    temp2 = [(key,value) for key, value in temp.items()]
    df = pd.DataFrame(temp2, columns = ['category','category_vec'])
    df.to_csv(os.path.join(args.output_folder, 'category_map.txt'),sep='\t',index=None)

    map_dict = id_map.set_index('curie')['id'].to_dict()

    graph_data = pd.concat([graph_data,graph_data.source.rename('id1').map(map_dict),graph_data.target.rename('id2').map(map_dict)],axis=1)
    graph_data = graph_data.sort_values(by=['id1','id2'])
    graph_data = graph_data.reset_index(drop=True)

    #use part of data as validation data
    id = [index - 1 for index in list(entity2id.values())]
    random.seed(args.seed)
    random.shuffle(id)
    valid = id[:int(len(entity2id)*args.validation_percent)]

    # generate Graph json file

    data = {'directed': False,
            'graph': {'name': 'disjoint_union(,)'},
            'nodes': [],
            'links': [],
            "multigraph": False
            }

    if args.process==-1:
        with multiprocessing.Pool() as executor:
            if emb_file is None:
                out_iters = [(node, False) for node in list(entity2id.values())]
            else:
                out_iters = [(node, True) for node in list(entity2id.values())]
            out_res = [elem for elem in chain.from_iterable(executor.map(initialize_node, out_iters))]

        data['nodes'] = out_res
    else:
        with multiprocessing.Pool(processes=args.process) as executor:
            if emb_file is None:
                out_iters = [(node, False) for node in list(entity2id.values())]
            else:
                out_iters = [(node, True) for node in list(entity2id.values())]
            out_res = [elem for elem in chain.from_iterable(executor.map(initialize_node, out_iters))]

        data['nodes'] = out_res

    if args.process == -1:
        with multiprocessing.Pool() as executor:
            out_iters = [index for index in range(graph_data.shape[0])]
            out_res = [elem for elem in chain.from_iterable(executor.map(initialize_edge, out_iters))]

        data['links'] = out_res
    else:
        with multiprocessing.Pool(processes=args.process) as executor:
            out_iters = [index for index in range(graph_data.shape[0])]
            out_res = [elem for elem in chain.from_iterable(executor.map(initialize_edge, out_iters))]

        data['links'] = out_res

    #save graph
    with open(os.path.join(args.output_folder, 'data-G.json'),'w') as f:
        f.write(json.dumps(data))

    #generate class_label
    class_map = {str(i-1):nodeclass_data_index[id2entity[i]][1] for i in list(entity2id.values())}
    #save labels
    with open(os.path.join(args.output_folder, 'data-class_map.json'),'w') as f:
        f.write(json.dumps(class_map))

    #generate id_label
    id_map = {str(i-1):(i-1) for i in list(entity2id.values())}
    #save nodes
    with open(os.path.join(args.output_folder, 'data-id_map.json'),'w') as f:
        f.write(json.dumps(id_map))

    #generate feats.npy
    if emb_file is not None:
        feats = np.array([emb_file[id2entity[i]] for i in list(entity2id.values())])
        with open(os.path.join(args.output_folder, 'data-feats.npy'), 'wb') as f:
            np.save(f, feats)
