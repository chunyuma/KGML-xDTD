import sys, os
from typing import List
import logging
import numpy as np
import pandas as pd
import pickle
import torch
from tqdm import tqdm
import argparse

pathlist = os.getcwd().split(os.path.sep)
Rootindex = pathlist.index("KGML-xDTD")
sys.path.append(os.path.sep.join([*pathlist[:(Rootindex + 1)], 'model_evaluation', 'scripts']))
import eval_utilities

class MOA_Mode:

    def __init__(self, args: argparse.Namespace, logger: logging.RootLogger):
        
        self.data_path = args.data_path
        self.model_path = args.model_path
        self.logger = logger

        ## set up additional variables
        self.results = dict()


    def _load_data(self, model: str):

        ## read drugmechdb matched paths
        with open(os.path.join(self.data_path, 'moa_evaluation_data', 'match_paths.pkl'), 'rb') as file_in:
            self.match_paths = pickle.load(file_in)

        ## read training drug-disease pairs
        train_pairs = pd.read_csv(os.path.join(self.data_path,'train_pairs.txt'), sep='\t', header=0)
        self.train_pairs_df = {(source, target):1 for (source, target) in train_pairs[['source','target']].to_numpy()}
        ## read validation drug-disease pairs
        val_pairs = pd.read_csv(os.path.join(self.data_path,'val_pairs.txt'), sep='\t', header=0)
        self.val_pairs_df = {(source, target):1 for (source, target) in val_pairs[['source','target']].to_numpy()}

        ## read precomputed path scores based on the specific model
        with open(os.path.join(self.data_path, 'moa_evaluation_data', model, 'res_all_paths_prob.pkl'), 'rb') as file_in:
            self.all_paths_prob = pickle.load(file_in)
        
        ## read mapping files
        relation2id, _ = eval_utilities.load_index(os.path.join(self.data_path, 'relation2freq.txt'))
        self.filter_edges = [relation2id[edge] for edge in ['biolink:related_to','biolink:biolink:part_of','biolink:coexists_with','biolink:contraindicated_for'] if relation2id.get(edge)]


    def _calculate_metrics(self):

        result = dict()

        ## sorted paths based on probability and store them in dictionary
        all_path_dict = dict()
        for (source, target) in tqdm(self.all_paths_prob):
            all_path_dict[(source, target)] = dict()
            try:
                edge_mat, node_mat, score_mat = self.all_paths_prob[(source, target)]
                temp = pd.DataFrame(edge_mat.numpy())
                keep_index = list(temp.loc[~(temp[1].isin(self.filter_edges) | temp[2].isin(self.filter_edges) | temp[3].isin(self.filter_edges)),:].index)
                edge_mat = edge_mat[keep_index]
                node_mat = node_mat[keep_index]
                score_mat = score_mat[keep_index]
                self.all_paths_prob[(source, target)] = [edge_mat, node_mat, score_mat]
            except:
                continue
            _, index_mat = torch.sort(score_mat, descending=True)
            edge_mat, node_mat, score_mat = edge_mat[index_mat], node_mat[index_mat], score_mat[index_mat]
            node_mat = node_mat.numpy()
            for index, x in enumerate(node_mat):
                if tuple(x) in all_path_dict[(source, target)]:
                    all_path_dict[(source, target)][tuple(x)].append([tuple(edge_mat[index].numpy()), score_mat[index].item(), index+1])
                else:
                    all_path_dict[(source, target)][tuple(x)] = []
                    all_path_dict[(source, target)][tuple(x)].append([tuple(edge_mat[index].numpy()), score_mat[index].item(), index+1])


        ## find the matched path indexes and store them in dictionary
        match_paths_dict = dict()
        for (source, target) in self.match_paths:
            temp = self.match_paths[(source, target)][1].unique(dim=0).numpy()
            match_paths_dict[(source, target)] = dict()
            for x in temp:
                test = all_path_dict[(source, target)][tuple(x)]
                match_paths_dict[(source, target)][tuple(x)] = (sorted(test, key=lambda row: row[2])[0], len(self.all_paths_prob[(source, target)][2]))

        ## calculate MPR 
        percentile = []
        for x in match_paths_dict:
            if x not in self.train_pairs_df and x not in self.val_pairs_df:
                temp = sorted(match_paths_dict[x].items(), key=lambda row: row[1][0][2])
                percentile += [1 - temp[0][1][0][2]/temp[0][1][1]]
        result['mpr'] = f"{np.array(percentile).mean():.3%}"

        ## calculate MRR 
        MRR = []
        for x in match_paths_dict:
            if x not in self.train_pairs_df and x not in self.val_pairs_df:
                temp = sorted(match_paths_dict[x].items(), key=lambda row: row[1][0][2])
                MRR += [1/temp[0][1][0][2]]
        result['mrr'] = f"{np.array(MRR).mean():.3f}"
        
        ## calculate Hit@K
        for k in [1, 10, 50, 100, 500]:
            hit_at_k = []
            count = []
            for index, x in enumerate(match_paths_dict):
                if x not in self.train_pairs_df and x not in self.val_pairs_df:
                    temp = sorted(match_paths_dict[x].items(), key=lambda row: row[1][0][2])
                    if temp[0][1][0][2] <= k:
                        hit_at_k += [index]
                    count += [index]
            result[f'Hit@{k}'] = f"{len(hit_at_k)/len(count):.3f}"

        return result

    def _report_evaluation(self):

        info = "\n"

        info += "Model\tMPR\tMRR\tHit@1\tHit@10\tHit@50\tHit@100\tHit@500\t\n"
        for model in ['multihop', 'kgml_xdtd_wo_dp', 'kgml_xdtd']:
            if model in self.results and len(self.results[model]) > 0:
                info += f"{model}\t{self.results[model]['mpr']}\t{self.results[model]['mrr']}\t{self.results[model]['Hit@1']}\t{self.results[model]['Hit@10']}\t{self.results[model]['Hit@50']}\t{self.results[model]['Hit@100']}\t{self.results[model]['Hit@500']}\n"

        self.logger.info(info)

    def do_evaluation(self, models: List[str]):

        if 'all' in models:
            models = ['multihop', 'kgml_xdtd_wo_dp', 'kgml_xdtd']

        for model in models:
            self.logger.info(f"Calculating MPR, MRR, Hit@K based on the score of paths generated by {model} model.")
            ## load data
            self._load_data(model)
            result = self._calculate_metrics()
            self.results.update({model: result})

        self._report_evaluation()