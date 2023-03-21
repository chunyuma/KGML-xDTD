## This script is used to generate random walk file (eg. walks.txt, please see https://github.com/williamleif/GraphSAGE
# for more details) via batch by batch for running Graphsage

from __future__ import print_function

import json
import numpy as np
import pandas as pd
import random
import os
import sys
import argparse
from networkx.readwrite import json_graph
import multiprocessing
from datetime import datetime
from itertools import chain
import utils

## setting functions compatible with parallel running
def run_random_walks(this):

    pairs = []
    node, num_walks, walk_len = this

    if G.degree(node) == 0:
        pairs = pairs
    else:
        for _ in range(num_walks):
            curr_node = node
            for _ in range(walk_len):
                neighbors = [n for n in G.neighbors(curr_node)]
                if len(neighbors) == 0:
                    ## no neighbor, stop BFS searching
                    break
                else:
                    next_node = random.choice()
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node, curr_node))
                curr_node = next_node
    return pairs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default="../log_folder")
    parser.add_argument("--log_name", type=str, help="log file name", default="step6_2.log")
    parser.add_argument("--Gjson", type=str, help="The path of G.json file")
    parser.add_argument("--walk_length", type=int, help="Random walk length", default=100)
    parser.add_argument("--number_of_walks", type=int, help="Number of random walks per node", default=10)
    parser.add_argument("--batch_size", type=int, help="Size of batch for each run", default=200000)
    parser.add_argument("--process", type=int, help="Number of processes to be used", default=-1)
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default="../data/graphsage_input")

    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir,args.log_name))
    logger.info(args)

    #create output directory
    if not os.path.isdir(args.output_folder):
        os.makedirs(args.output_folder)

    #read the graph file
    with open(args.Gjson,'r') as input_file:
	    G_data = json.load(input_file)

    # transform to networkx graph format
    G = json_graph.node_link_graph(G_data)
    # pull out the training nodes and generate the training subgraph
    G_nodes = [n for n in G.nodes() if not G.nodes[n]["val"] and not G.nodes[n]["test"]]
    G = G.subgraph(G_nodes)
    del G_data ## delete variable to release ram

    # set up the batches
    batch =list(range(0,len(G_nodes),args.batch_size))
    batch.append(len(G_nodes))

    logger.info(f'Total training data: {len(G_nodes)}')
    logger.info(f'The number of nodes in training graph: {len(G.nodes)}')

    logger.info(f'total batch: {len(batch)-1}')

    ## run each batch in parallel
    for i in range(len(batch)):
        if((i+1)<len(batch)):
            logger.info(f'Here is batch{i+1}')
            start = batch[i]
            end = batch[i+1]
            if args.process == -1:
                with multiprocessing.Pool() as executor:
                    out_iters = [(node, args.number_of_walks, args.walk_length) for node in G_nodes[start:end]]
                    out_res = [elem for elem in chain.from_iterable(executor.map(run_random_walks, out_iters))]
            else:
                with multiprocessing.Pool(processes=args.process) as executor:
                    out_iters = [(node, args.number_of_walks, args.walk_length) for node in G_nodes[start:end]]
                    out_res = [elem for elem in chain.from_iterable(executor.map(run_random_walks, out_iters))]
            with open(os.path.join(args.output_folder, 'data-walks.txt'), "a") as fp:
                if i==0:
                    fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in out_res]))
                else:
                    fp.write("\n")
                    fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in out_res]))

