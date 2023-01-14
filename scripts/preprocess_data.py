
import argparse
import collections
import numpy as np
import pandas as pd
import os
import pickle
import utils
from tqdm import tqdm
import networkx as nx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default="../log_folder")
    parser.add_argument("--log_name", type=str, help="log file name", default="step3_1.log")
    parser.add_argument('--data_dir', type=str, help='Full path of data folder', default='../data')
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default="../data")
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir,args.log_name))
    logger.info(args)

    # Create entity and predicate indices
    entity_hist = collections.defaultdict(int)
    predicate_hist = collections.defaultdict(int)
    type_hist = collections.defaultdict(int)

    all_graph_nodes_info = pd.read_csv(os.path.join(args.data_dir, 'all_graph_nodes_info.txt'), sep='\t', header=0)
    entity_to_type = {all_graph_nodes_info.loc[index, 'id']:all_graph_nodes_info.loc[index, 'category'] for index in range(len(all_graph_nodes_info))}

    with open(os.path.join(args.data_dir, 'graph_edges.txt'), 'r') as f:
        graph_edge_triples = [l.strip() for index, l in enumerate(f.readlines()) if index!=0]
    graph_edge_triples = set(graph_edge_triples)

    # Index entities and predicates
    for line in graph_edge_triples:
        source, target, predicate, _ = line.strip().split('\t')
        entity_hist[source] += 1
        entity_hist[target] += 1
        predicate_hist[predicate] += 1
        type_hist[entity_to_type[source]] += 1
        type_hist[entity_to_type[target]] += 1

    ## add additional edge type for true positive and true negative for drug-disease pairs
    predicate_hist['biolink:has_effect'] += 1
    predicate_hist['biolink:has_no_effect'] += 1

    # Save the entity and predicate indices sorted by decreasing frequency
    with open(os.path.join(args.output_folder, 'entity2freq.txt'), 'w') as outfile:
        outfile.write(f"{utils.DUMMY_ENTITY}\t{1}\n")
        for entity, freq in utils.hist_to_vocab(entity_hist):
            outfile.write(f"{entity}\t{freq}\n")

    with open(os.path.join(args.output_folder, 'relation2freq.txt'), 'w') as outfile:
        outfile.write(f"{utils.DUMMY_RELATION}\t{1}\n")
        outfile.write(f"{utils.SELF_LOOP_RELATION}\t{1}\n")
        for predicate, freq in utils.hist_to_vocab(predicate_hist):
            outfile.write(f"{predicate}\t{freq}\n")

    with open(os.path.join(args.output_folder, 'type2freq.txt'), 'w') as outfile:
        for e_type, freq in utils.hist_to_vocab(type_hist):
            outfile.write(f"{e_type}\t{freq}\n")

    logger.info(f'{len(entity_hist)} entities indexed')
    logger.info(f'{len(predicate_hist)} relations indexed')
    logger.info(f'{len(type_hist)} types indexed')

    entity2id, id2entity = utils.load_index(os.path.join(args.data_dir, 'entity2freq.txt'))
    relation2id, id2relation = utils.load_index(os.path.join(args.data_dir, 'relation2freq.txt'))
    type2id, id2type = utils.load_index(os.path.join(args.data_dir, 'type2freq.txt'))

    ## save train edge triples as a adjacency list
    adj_list = collections.defaultdict(collections.defaultdict)
    entity2typeid = [-1 for i in range(len(entity2id))]
    num_triples = 0
    for line in tqdm(graph_edge_triples):
        source, target, predicate, _ = line.strip().split('\t')
        entity2typeid[entity2id[source]] = type2id[entity_to_type[source]]
        entity2typeid[entity2id[target]] = type2id[entity_to_type[target]]
        if not relation2id[predicate] in adj_list[entity2id[source]]:
            adj_list[entity2id[source]][relation2id[predicate]] = set()
        if entity2id[target] in adj_list[entity2id[source]][relation2id[predicate]]:
            logger.info('Duplicate triple found: {} ({}, {}, {})!'.format(line.strip(), source, predicate, target))
        else:
            adj_list[entity2id[source]][relation2id[predicate]].add(entity2id[target])
        num_triples += 1

    logger.info(f'{num_triples} triples processed')

    # Save adjacency list
    with open(os.path.join(args.output_folder, 'adj_list.pkl'), 'wb') as outfile:
        pickle.dump(dict(adj_list), outfile)
    with open(os.path.join(args.output_folder, 'entity2typeid.pkl'), 'wb') as outfile:
        pickle.dump(entity2typeid, outfile)

    # Calculate PageRank for each vertice
    G = nx.DiGraph()
    all_edges = []
    for source in adj_list:
        temp_edges = [(source,target) for key in adj_list[source] for target in adj_list[source][key]]
        all_edges += temp_edges
    all_edges = list(set(all_edges))
    G.add_edges_from(all_edges)
    pr = nx.pagerank(G)

    with open(os.path.join(args.output_folder, 'kg.pgrk'), 'w') as outfile:
        itemlist = [f"{id2entity[curie]}\t{value}" for curie,value in pr.items()]
        outfile.write("\n".join(itemlist))
