import sys, os
from typing import List
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import torch
from sklearn.metrics import f1_score
from torch_geometric.data import NeighborSampler
import pickle
import argparse

pathlist = os.getcwd().split(os.path.sep)
Rootindex = pathlist.index("KGML-xDTD")
sys.path.append(os.path.sep.join([*pathlist[:(Rootindex + 1)], 'model_evaluation', 'scripts']))
import eval_utilities
from graphsage_link.model import GraphSage
from graphsage_link.dataset import Dataset
sys.path.append(os.path.sep.join([*pathlist[:(Rootindex + 1)], 'openke']))
from module.model import TransE, RotatE, DistMult, ComplEx, TransR, SimplE, Analogy

class DDP_Mode:

    def __init__(self, args: argparse.Namespace, logger: logging.RootLogger):
        
        self.data_path = args.data_path
        self.model_path = args.model_path
        self.logger = logger

        self.use_gpu = args.use_gpu
        self.device = args.device
        if self.use_gpu:
            torch.cuda.set_device(self.device)
        self.batch_size = args.batch_size

        ## set up additional variables
        self.results = dict()

    def _load_data(self, use_random: bool=True):

        if use_random:
            # self.test_data = pd.read_csv(os.path.join(self.data_path, 'test_pairs.txt'), sep='\t', header=0)
            self.random_data = pd.read_csv(os.path.join(self.data_path, 'random_pairs.txt'), sep='\t', header=0)
            self.random_data_list = []
            ten_times_random_data_dir = os.path.join(self.data_path, '10times_random_pairs')
            for random_data_file in sorted(os.listdir(ten_times_random_data_dir), key=lambda x: int(x.replace('.txt','').split('_')[-1])):
                self.random_data_list += [pd.read_csv(os.path.join(ten_times_random_data_dir, random_data_file), sep='\t', header=0)]
        else:
            self.train_data = pd.read_csv(os.path.join(self.data_path, 'train_pairs.txt'), sep='\t', header=0)
            self.val_data = pd.read_csv(os.path.join(self.data_path, 'val_pairs.txt'), sep='\t', header=0)
            self.test_data = pd.read_csv(os.path.join(self.data_path, 'test_pairs.txt'), sep='\t', header=0)
            temp = pd.concat([self.train_data,self.val_data,self.test_data]).reset_index(drop=True)
            temp = temp.loc[temp['y'] == 1,['source','target']].reset_index(drop=True)

            ## get all true positive pairs
            self.all_tp_pairs_dict = {"_".join(pair):1 for pair in temp.to_numpy()}

        ## get all drugs and all diseases
        self.entity2id, self.id2entity = eval_utilities.load_index(os.path.join(self.data_path, 'entity2freq.txt'))
        self.relation2id, self.id2relation = eval_utilities.load_index(os.path.join(self.data_path, 'relation2freq.txt'))
        type2id, _ = eval_utilities.load_index(os.path.join(self.data_path, 'type2freq.txt'))
        with open(os.path.join(self.data_path, 'entity2typeid.pkl'), 'rb') as infile:
            entity2typeid = pickle.load(infile)
        drug_type = ['biolink:Drug', 'biolink:SmallMolecule']
        drug_type_ids = [type2id[x] for x in drug_type]
        self.all_drug_ids = [self.id2entity[index] for index, typeid in enumerate(entity2typeid) if typeid in drug_type_ids]
        disease_type = ['biolink:Disease', 'biolink:PhenotypicFeature', 'biolink:BehavioralFeature', 'biolink:DiseaseOrPhenotypicFeature']
        disease_type_ids = [type2id[x] for x in disease_type]
        self.all_disease_ids = [self.id2entity[index] for index, typeid in enumerate(entity2typeid) if typeid in disease_type_ids]


    def _load_embedding(self, model: str):
        
        if model in ['graphsage_logistic', 'graphsage_svm', '2class_kgml_xdtd', 'kgml_xdtd']:
            return eval_utilities.load_graphsage_unsupervised_embeddings(self.data_path, True)

        if model in ['kgml_xdtd_wo_naes']:
            return eval_utilities.load_graphsage_unsupervised_embeddings(self.data_path, False)

        if model in ['graphsage_link']:
            return eval_utilities.load_biobert_embeddings(self.data_path)

    def _load_model(self, model: str):

        if model in ['graphsage_logistic', 'graphsage_svm', '2class_kgml_xdtd', 'kgml_xdtd_wo_naes', 'kgml_xdtd']:
            if model == 'kgml_xdtd':
                file_path = os.path.join(self.model_path, model, 'drp_module', 'model.pt')
            else:
                file_path = os.path.join(self.model_path, model, 'model.pt')
            return joblib.load(file_path)

        if model in ['graphsage_link']:
            fitModel = GraphSage(100, 256, 1, self.use_gpu, self.device)
            fitModel.load_state_dict(torch.load(os.path.join(self.model_path, model, 'model.pt'), map_location=self.device)['model_state_dict'])
            fitModel.eval()
            return fitModel

        if model in ['transe', 'transr', 'rotate', 'distmult', 'complex', 'analogy', 'simple']:
            model2dim = {
                'transe': 100,
                'distmult': 100,
                'simple': 100,
                'rotate': 30,
                'complex': 50,
                'transr': 50,
                'analogy': 20,
            }
            dim = model2dim[model]
            
            model_dir = os.path.join(self.model_path, model, 'model.pt')

            if model == "transe":
                kg_model = TransE(
                    ent_tot = self.num_entity,
                    rel_tot = self.num_relation,
                    dim = dim, 
                    p_norm = 1, 
                    norm_flag = True)
            elif model == "rotate":
                kg_model = RotatE(
                ent_tot = self.num_entity,
                rel_tot = self.num_relation,
                dim = dim,
                margin = 6.0,
                epsilon = 2.0,
            )
            elif model == "complex":
                kg_model = ComplEx(
                ent_tot = self.num_entity,
                rel_tot = self.num_relation,
                dim = dim)
            elif model == 'distmult':
                kg_model = DistMult(
                    ent_tot = self.num_entity,
                    rel_tot = self.num_relation,
                    dim = dim
                )
            elif model == 'transr':
                kg_model = TransR(
                    ent_tot = self.num_entity,
                    rel_tot = self.num_relation,
                    dim_e = dim,
                    dim_r = dim,
                    p_norm = 1, 
                    norm_flag = True,
                    rand_init = True)
            elif model == 'simple':
                kg_model = SimplE(
                    ent_tot = self.num_entity,
                    rel_tot = self.num_relation,
                    dim = dim
                )
            elif model == 'analogy':
                kg_model = Analogy(
                    ent_tot = self.num_entity,
                    rel_tot = self.num_relation,
                    dim = dim
                )
            else:
                raise ValueError("Please choose model_type from [transe,transr,rotate,distmult,complex,analogy,simple]")

            kg_model.load_checkpoint(model_dir)

            return kg_model

    def _generate_X_and_y(self, embeddings: dict, data_df: pd.core.frame.DataFrame):

        X = np.vstack([np.hstack([embeddings[data_df.loc[index,'source']], embeddings[data_df.loc[index,'target']]]) for index in range(len(data_df))])
        y = np.array(list(data_df['y']))

        return X, y

    def _process_data_for_kgembedding(self, use_random: bool):

        self.test_data = self._transfer_name_to_id(self.test_data)
        
        if use_random:
            self.random_data_list = [self._transfer_name_to_id(random_data) for random_data in self.random_data_list]
            

    def _transfer_name_to_id(self, data: pd.core.frame.DataFrame):

        # kg_data_path = os.path.join(self.data_path, "kgembedding_data")

        # with open(os.path.join(kg_data_path, "entity2id.txt"), "r") as f:
        #     entity2id_raw = f.readlines()
        # with open(os.path.join(kg_data_path, "relation2id.txt"), "r") as f:
        #     relation2id_raw = f.readlines()

        # self.entity2id = {}
        # self.relation2id = {}
        # for line in entity2id_raw[1:]:
        #     line = line.strip("\n").split("\t")
        #     self.entity2id[line[0]] = line[1]
        # for line in relation2id_raw[1:]:
        #     line = line.strip("\n").split("\t")
        #     self.relation2id[line[0]] = line[1]
        
        tpID = self.relation2id['biolink:has_effect']
        tnID = self.relation2id['biolink:has_no_effect']

        data["source_id"] = [self.entity2id[source] for source in data["source"]]
        data["target_id"] = [self.entity2id[target] for target in data["target"]]
        data["relation_id"] = [tpID if int(relation) == 1 else tnID for relation in data["y"]]

        data["source_id"], data["target_id"], data["relation_id"] = data["source_id"].astype("float"), data["target_id"].astype("float"), data["relation_id"].astype("float")
        self.num_entity = len(self.entity2id)
        self.num_relation = len(self.relation2id)

        return data

    def _kgembedding_models_tptn_probability(self, sources, targets, kg_model):
        kg_model = kg_model.to(self.device)
        num_sample = 2 * len(sources) 
        data = {}
        data['batch_h'] = torch.zeros([num_sample])
        data['batch_t'] = torch.zeros([num_sample])
        data['batch_r'] = torch.zeros([num_sample]) 
        data['batch_r'][:len(sources)] = int(self.relation2id['biolink:has_effect'])
        data['batch_r'][len(sources):] = int(self.relation2id['biolink:has_no_effect'])
        data['mode'] = 'normal'
        for i, (source, target) in enumerate(zip(sources, targets)):
            s_id = source
            t_id = target
            data['batch_h'][i] = float(s_id)
            data['batch_h'][i+len(sources)] = float(s_id)
            data['batch_t'][i] = float(t_id)
            data['batch_t'][i+len(sources)] = float(t_id)
        
        data['batch_h'] = data['batch_h'].long().to(self.device)
        data['batch_t'] = data['batch_t'].long().to(self.device)
        data['batch_r'] = data['batch_r'].long().to(self.device)
        
        return kg_model.predict(data)

    def _kgembedding_models_tp_probability(self, sources, targets, kg_model):
        kg_model = kg_model.to(self.device)
        data = {}
        data['batch_h'] = torch.Tensor(sources)
        data['batch_t'] = torch.Tensor(targets)
        data['batch_r'] = torch.ones([len(sources)]) * int(self.relation2id["biolink:has_effect"])
        data['mode'] = 'normal'

        data['batch_h'] = data['batch_h'].long().to(self.device)
        data['batch_t'] = data['batch_t'].long().to(self.device)
        data['batch_r'] = data['batch_r'].long().to(self.device)
        
        return kg_model.predict(data)

    @staticmethod
    def calculate_accuracy(preds: np.ndarray, labels: np.ndarray):

        return (preds == labels).astype(float).mean()

    @staticmethod
    def calculate_f1_score(preds: np.ndarray, labels: np.ndarray, average: str='macro'):

        if average in ['binary', 'micro', 'macro', 'weighted']:
            f1score = f1_score(labels, preds, average=average)
            return f1score
        else:
            return None

    @staticmethod
    def calculate_rank(true_pairs: pd.core.frame.DataFrame, random_pairs: pd.core.frame.DataFrame):

        ranks = []
        Q_n = len(true_pairs)
        for index in range(Q_n):
            query_drug = true_pairs['source'][index]
            query_disease = true_pairs['target'][index]
            this_query_score = true_pairs['prob'][index]
            all_random_probs_for_this_query = list(random_pairs.loc[random_pairs['source'].isin([query_drug]),'prob'])
            all_random_probs_for_this_query += list(random_pairs.loc[random_pairs['target'].isin([query_disease]),'prob'])
            all_random_probs_for_this_query = all_random_probs_for_this_query[:1000]
            all_in_list = [this_query_score] + all_random_probs_for_this_query
            rank = list(torch.tensor(all_in_list).sort(descending=True).indices.numpy()).index(0)+1
            ranks += [rank]
            
        return ranks

    def calculate_rank_all(self, data_pos_df, all_drug_ids, all_disease_ids, entity_embeddings, all_tp_pairs_dict, fitModel, is_kgembedding_model = False, is_graphsage_link = False, data = None):
        rank_dict = dict()
        total = data_pos_df.shape[0]

        if is_graphsage_link:
            for _, n_id, adjs in NeighborSampler(data.edge_index, sizes=[96, 96], batch_size=data.feat.shape[0], shuffle=False, num_workers=16):
                adjs = [(adj.edge_index,adj.size) for adj in adjs]
                n_id = n_id.to(self.device)
                x_n_id = data.feat[n_id]
                all_embeddings = fitModel.get_gnn_embedding(x_n_id, adjs, n_id)
                break

        for index, (source, target) in enumerate(data_pos_df[['source','target']].to_numpy()):
            print(f"calculating rank {index+1}/{total}", flush=True)
            this_pair = source + '_' + target
            
            if is_graphsage_link:
                temp_df = pd.concat([pd.DataFrame(zip(all_drug_ids,[target]*len(all_drug_ids))),pd.DataFrame(zip([source]*len(all_disease_ids),all_disease_ids))]).reset_index(drop=True)
                link =temp_df[[0,1]].apply(lambda row: [self.entity2id.get(row[0]) - 1, self.entity2id.get(row[1]) - 1], axis=1, result_type='expand').rename(columns={0: "source", 1: "target"})
                link = torch.tensor(np.array(link), dtype=torch.long).to(self.device)
                temp_df[2] = temp_df[0] + '_' + temp_df[1]
                temp_df[3] = fitModel.predict2(all_embeddings, link, n_id)
            elif is_kgembedding_model:
                all_drugs = [float(self.entity2id[d]) for d in all_drug_ids]
                all_diseases = [float(self.entity2id[d]) for d in all_disease_ids]
                all_sources = all_drugs + [data_pos_df.iloc[index]['source_id']] * len(all_diseases)
                all_targets = [data_pos_df.iloc[index]['target_id']] * len(all_drugs) + all_diseases
                pred_probs = -self._kgembedding_models_tp_probability(all_sources, all_targets, fitModel)
                temp_df = pd.concat([pd.DataFrame(zip(all_drug_ids,[target]*len(all_drug_ids))),pd.DataFrame(zip([source]*len(all_disease_ids),all_disease_ids))]).reset_index(drop=True)
                temp_df[2] = temp_df[0] + '_' + temp_df[1]
                temp_df[3] = pred_probs
            else:
                X_drug = np.vstack([np.hstack([entity_embeddings[drug_id],entity_embeddings[target]]) for drug_id in all_drug_ids])
                X_disease = np.vstack([np.hstack([entity_embeddings[source],entity_embeddings[disease_id]]) for disease_id in all_disease_ids])
                all_X = np.concatenate([X_drug,X_disease],axis=0)
                pred_probs = fitModel.predict_proba(all_X)
                temp_df = pd.concat([pd.DataFrame(zip(all_drug_ids,[target]*len(all_drug_ids))),pd.DataFrame(zip([source]*len(all_disease_ids),all_disease_ids))]).reset_index(drop=True)
                temp_df[2] = temp_df[0] + '_' + temp_df[1]
                temp_df[3] = pred_probs[:,1]

            this_row = temp_df.loc[temp_df[2]==this_pair,:].reset_index(drop=True).iloc[[0]]
            temp_df = temp_df.loc[temp_df[2]!=this_pair,:].reset_index(drop=True)

            ## filter 
            temp_df = temp_df.loc[~temp_df[2].isin(list(all_tp_pairs_dict.keys())),:].reset_index(drop=True)
            # (1) for drug replacement
            temp_df_1 = pd.concat([temp_df.loc[temp_df[1] == target,:],this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            drug_rank = temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1
            # (2) for disease replacement
            temp_df_1 = pd.concat([temp_df.loc[temp_df[0] == source,:],this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            disease_rank = temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1
            # (3) for both replacement
            temp_df_1 = pd.concat([temp_df,this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            both_rank = temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1
            rank_dict[(source, target)] = (drug_rank, disease_rank, both_rank)

        return rank_dict


    def _report_evaluation(self, use_random: bool=True):

        info = "\n"

        if use_random:
            info += "Model\tAccuracy\tMacro F1 score\tMRR\tHit@1\tHit@3\tHit@5\n"
            for model in ["transe","transr","rotate","distmult","complex","analogy","simple","graphsage_link","graphsage_logistic","graphsage_svm","kgml_xdtd_wo_naes","2class_kgml_xdtd","kgml_xdtd"]:
                if model in self.results and len(self.results[model]) > 0:
                    info += f"{model}\t{self.results[model]['accuracy']}\t{self.results[model]['macro_f1score']}\t{self.results[model]['mrr']}\t{self.results[model]['hit@1']}\t{self.results[model]['hit@3']}\t{self.results[model]['hit@5']}\n"
        else:
            info += "Model\tMRR\tHit@1\tHit@3\tHit@5\n"
            for model in ["transe","transr","rotate","distmult","complex","analogy","simple","graphsage_link","graphsage_logistic","graphsage_svm","kgml_xdtd_wo_naes","2class_kgml_xdtd","kgml_xdtd"]:
                if model in self.results and len(self.results[model]) > 0:
                    for replacement_type in ['drug', 'disease', 'both']:
                        info += f"{replacement_type.capitalize()} Replacement: {model}\t{self.results[model][f'{replacement_type}_mrr']}\t{self.results[model][f'{replacement_type}_hit@1']}\t{self.results[model][f'{replacement_type}_hit@3']}\t{self.results[model][f'{replacement_type}_hit@5']}\n"

        self.logger.info(info)

    def do_evaluation(self, models: List[str], use_random: bool=True):

        if 'all' in models:
            models = ["transe","transr","rotate","distmult","complex","analogy","simple","graphsage_link","graphsage_logistic","graphsage_svm","kgml_xdtd_wo_naes","2class_kgml_xdtd","kgml_xdtd"]

        ## load data
        self._load_data(use_random)

        for model in list(set(models).intersection(set(["transe","transr","rotate","distmult","complex","analogy","simple"]))):
            result = self.eval_kgembedding_models(model_type=model, use_random=use_random)
            self.results.update({model: result})

        if "graphsage_link" in models:
            result = self.eval_graphsage_link(use_random)
            self.results.update({'graphsage_link': result})

        for model in list(set(models).intersection(set(["graphsage_logistic","graphsage_svm","kgml_xdtd_wo_naes","2class_kgml_xdtd","kgml_xdtd"]))):
            result = self.eval_rest_models(model_type=model, use_random=use_random)
            self.results.update({model: result})

        self._report_evaluation(use_random)

    def eval_kgembedding_models(self, model_type: str, use_random: bool):

        self.logger.info(f"Evaluating {model_type} model ....")
        
        result = {}

        ## load data and process it
        self._process_data_for_kgembedding(use_random)

        ## load model
        kg_model = self._load_model(model_type)

        test_data = self.test_data
        test_data = test_data[test_data["y"] != 2]

        if use_random:
            self.logger.info("Using 1,000 random drug-disease pairs for MRR and Hit@K calculation")

            acc_res = self._kgembedding_models_tptn_probability(test_data["source_id"], test_data["target_id"], kg_model)
            num_samples = len(acc_res) // 2
            scores = np.zeros([num_samples, 2])
            scores[:,0] = acc_res[:num_samples]
            scores[:,1] = acc_res[num_samples:]

            test_preds = np.zeros([num_samples])
            for i in range(num_samples):
                test_preds[i] = 1 if scores[i][0] < scores[i][1] else 0
            test_scores = scores[:,0]
            test_labels = test_data["y"].to_numpy()

            acc = self.calculate_accuracy(test_preds, test_labels)
            macro_f1score = self.calculate_f1_score(test_preds, test_labels)

            result['accuracy'] = f"{acc:.3f}"
            result['macro_f1score'] = f"{macro_f1score:.3f}"

            test_data['prob'] = -test_scores
            true_pairs = test_data.loc[test_data['y']==1,:].reset_index(drop=True)

            mrr_res = []
            hit1_res = []
            hit3_res = []
            hit5_res = []

            for random_data in tqdm(self.random_data_list, desc="using 10 Random Data"):
                random_scores = self._kgembedding_models_tp_probability(random_data["source_id"], random_data["target_id"], kg_model)
                random_data['prob'] = -random_scores
                random_pairs = random_data
                ranks = self.calculate_rank(true_pairs, random_pairs)
                mrr_res.append(np.array([1/rank for rank in ranks]).mean())
                hit1_res.append(np.array([x <= 1 for x in ranks]).mean())
                hit3_res.append(np.array([x <= 3 for x in ranks]).mean())
                hit5_res.append(np.array([x <= 5 for x in ranks]).mean())

            result['mrr'] = f"{np.array(mrr_res).mean():.3f} (+/- {np.array(mrr_res).std():.3f})"
            for k in [1,3,5]:
                res = eval(f"hit{k}_res")
                result[f'hit@{k}'] = f"{np.array(res).mean():.3f} (+/- {np.array(res).std():.3f})"
        
        else:
            self.logger.info("Using three replacement methods for MRR and Hit@K calculation")
            test_data_pos = test_data.loc[test_data['y'] == 1,:].reset_index(drop=True)
            rank_res = self.calculate_rank_all(data_pos_df=test_data_pos, all_drug_ids=self.all_drug_ids, all_disease_ids=self.all_disease_ids, entity_embeddings=None, all_tp_pairs_dict=self.all_tp_pairs_dict, fitModel=kg_model, is_kgembedding_model=True)

            drug_mrr = np.array([1/rank_res[i][0] for i in rank_res]).mean()
            result['drug_mrr'] = f"{drug_mrr:.8f}"
            disease_mrr = np.array([1/rank_res[i][1] for i in rank_res]).mean()
            result['disease_mrr'] = f"{disease_mrr:.8f}"
            both_mrr = np.array([1/rank_res[i][2] for i in rank_res]).mean()
            result['both_mrr'] = f"{both_mrr:.8f}"

            for k in [1,3,5]:
                drug_hitk = np.array([True if rank_res[i][0] <= k else False for i in rank_res]).astype(float).mean()
                result[f'drug_hit@{k}'] = f"{drug_hitk:.8f}"
                disease_hitk = np.array([True if rank_res[i][1] <= k else False for i in rank_res]).astype(float).mean()
                result[f'disease_hit@{k}'] = f"{disease_hitk:.8f}"
                both_hitk = np.array([True if rank_res[i][2] <= k else False for i in rank_res]).astype(float).mean()
                result[f'both_hit@{k}'] = f"{both_hitk:.8f}"
        
        return result

    def eval_graphsage_link(self, use_random: bool=True):
        
        def _generate_probas_and_label(model, loader, batch_data, init_emb, device):
            probas = []
            label = []

            with torch.no_grad():
                for batch_idx, (n_id, adjs) in enumerate(tqdm(loader)):
                    n_id = n_id[0].to(device)
                    adjs = [(adj[0][0],(int(adj[1][0]),int(adj[1][1]))) for adj in adjs]
                    link = batch_data[batch_idx][['source','target']].apply(lambda row: [self.entity2id.get(row[0]) - 1, self.entity2id.get(row[1]) - 1], axis=1, result_type='expand').rename(columns={0: "source", 1: "target"})
                    link = torch.tensor(np.array(link), dtype=torch.long).to(device)
                    x_n_id = init_emb[n_id]

                    pred = model.predict1(x_n_id, adjs, link, n_id).numpy()
                    probas.append(pred)
                    label.append(np.array(batch_data[batch_idx]['y']))

            probas = np.concatenate(probas)
            label = np.hstack(label)

            return probas, label


        self.logger.info("Evaluating graphsage_link model ....")

        result = {}

        ## load embedding
        entity_embeddings = self._load_embedding('graphsage_link')

        ## load model
        model = self._load_model('graphsage_link')

        ## load processed data
        dataset = Dataset(self.data_path, self.logger, entity_embeddings)
        data = dataset.get_dataset()
        _, _, test_batch, random_batch_list = dataset.get_train_val_test_random()
        test_loader = dataset.get_test_loader()
        init_emb = data.feat
        test_data = pd.concat(test_batch)

        if use_random:
            self.logger.info("Using 1,000 random drug-disease pairs for MRR and Hit@K calculation")
            
            ## calculate probability for test data
            test_probas, test_labels = _generate_probas_and_label(model, test_loader, test_batch, init_emb, self.device)
            # check_pairs = {(x[0],x[1]):1 for x in self.test_data.to_numpy() if x[2] != 2}
            # saved_index = [index for index, (source, target, y) in enumerate(test_data.to_numpy()) if (source, target) in check_pairs]
            # test_data = test_data.reset_index(drop=True).loc[saved_index,:].reset_index(drop=True)
            # test_probas = test_probas[saved_index]
            # test_labels = test_labels[saved_index]
            test_preds = np.argmax(test_probas, axis=1)

            acc = self.calculate_accuracy(test_preds, test_labels)
            macro_f1score = self.calculate_f1_score(test_preds, test_labels)

            result['accuracy'] = f"{acc:.3f}"
            result['macro_f1score'] = f"{macro_f1score:.3f}"

            ## calculate MRR and Hit@k
            test_data['prob'] = test_probas[:,1]
            true_pairs = test_data.loc[test_data['y']==1,:].reset_index(drop=True)

            mrr_res = []
            hit1_res = []
            hit3_res = []
            hit5_res = []

            for index, random_batch in enumerate(tqdm(random_batch_list, desc="using 10 Random Data")):
                random_loader = dataset.get_random_loader(index)
                random_data = pd.concat(random_batch)
                random_probas, _ = _generate_probas_and_label(model, random_loader, random_batch, init_emb, self.device)
                random_data['prob'] = random_probas[:,1]
                random_pairs = random_data

                ranks = self.calculate_rank(true_pairs, random_pairs)
                mrr_res.append(np.array([1/rank for rank in ranks]).mean())
                hit1_res.append(np.array([x <= 1 for x in ranks]).mean())
                hit3_res.append(np.array([x <= 3 for x in ranks]).mean())
                hit5_res.append(np.array([x <= 5 for x in ranks]).mean())

            result['mrr'] = f"{np.array(mrr_res).mean():.3f} (+/- {np.array(mrr_res).std():.3f})"
            for k in [1,3,5]:
                res = eval(f"hit{k}_res")
                result[f'hit@{k}'] = f"{np.array(res).mean():.3f} (+/- {np.array(res).std():.3f})"

        else:
            self.logger.info("Using three replacement methods for MRR and Hit@K calculation")
            self.logger.warning("The 'graphsage_link' model might need to take a few days to finish the evaluation")

            # use three replacement methods: drug-rank-based replacement, disease-rank-based replacement, combined Replacement
            test_data_pos = test_data.loc[test_data['y'] == 1,:].reset_index(drop=True)
            rank_res = self.calculate_rank_all(data_pos_df=test_data_pos, all_drug_ids=self.all_drug_ids, all_disease_ids=self.all_disease_ids, entity_embeddings=None, all_tp_pairs_dict=self.all_tp_pairs_dict, fitModel=model, is_graphsage_link=True, data=data)

            ##### filtering other tp pairs #####
            drug_mrr = np.array([1/rank_res[i][0] for i in rank_res]).mean()
            result['drug_mrr'] = f"{drug_mrr:.8f}"
            disease_mrr = np.array([1/rank_res[i][1] for i in rank_res]).mean()
            result['disease_mrr'] = f"{disease_mrr:.8f}"
            both_mrr = np.array([1/rank_res[i][2] for i in rank_res]).mean()
            result['both_mrr'] = f"{both_mrr:.8f}"

            for k in [1,3,5]:
                drug_hitk = np.array([True if rank_res[i][0] <= k else False for i in rank_res]).astype(float).mean()
                result[f'drug_hit@{k}'] = f"{drug_hitk:.8f}"
                disease_hitk = np.array([True if rank_res[i][1] <= k else False for i in rank_res]).astype(float).mean()
                result[f'disease_hit@{k}'] = f"{disease_hitk:.8f}"
                both_hitk = np.array([True if rank_res[i][2] <= k else False for i in rank_res]).astype(float).mean()
                result[f'both_hit@{k}'] = f"{both_hitk:.8f}"

        return result


    def eval_rest_models(self, model_type: str, use_random: bool=True):

        if model_type == "graphsage_svm":
            self.logger.info("Evaluating graphsage_svm model ....")
            self.logger.warning("The 'graphsage_svm' model might need to take a few hours to finish the evaluation")
        else:
            self.logger.info(f"Evaluating {model_type} model ....")

        result = {}

        ## load embedding
        entity_embeddings = self._load_embedding(model_type)

        ## load model
        model = self._load_model(model_type)

        if model_type not in ['kgml_xdtd_wo_naes','kgml_xdtd']:
            test_data = self.test_data.loc[self.test_data['y'] != 2,:].reset_index(drop=True)
        else:
            test_data = self.test_data
        test_X, test_labels = self._generate_X_and_y(entity_embeddings, test_data)

        if use_random:
            self.logger.info("Using 1,000 random drug-disease pairs for MRR and Hit@K calculation")

            ## calculate probability for test data
            test_probas = model.predict_proba(test_X)
            test_preds = np.argmax(test_probas, axis=1)

            acc = self.calculate_accuracy(test_preds, test_labels)
            macro_f1score = self.calculate_f1_score(test_preds, test_labels)

            if model_type in ['kgml_xdtd_wo_naes','kgml_xdtd']:
                ## scaled to 2class classification
                acc_2class = self.calculate_accuracy(np.argmax(test_probas[test_data['y']!=2][:,:2], axis=1), test_labels[test_data['y']!=2])
                macro_f1score_2class = self.calculate_f1_score(np.argmax(test_probas[test_data['y']!=2][:,:2], axis=1), test_labels[test_data['y']!=2])

                result['accuracy'] = f"{acc:.3f} ({acc_2class:.3f}*)"
                result['macro_f1score'] = f"{macro_f1score:.3f} ({macro_f1score_2class:.3f}*)"
            else:
                acc = self.calculate_accuracy(test_preds, test_labels)
                macro_f1score = self.calculate_f1_score(test_preds, test_labels)

                result['accuracy'] = f"{acc:.3f}"
                result['macro_f1score'] = f"{macro_f1score:.3f}"

            ## calculate MRR and Hit@k
            test_data['prob'] = test_probas[:,1]
            true_pairs = test_data.loc[test_data['y']==1,:].reset_index(drop=True)

            mrr_res = []
            hit1_res = []
            hit3_res = []
            hit5_res = []

            for random_data in tqdm(self.random_data_list, desc="using 10 Random Data"):
                random_X, _ = self._generate_X_and_y(entity_embeddings, random_data)
                ## calculate probability
                random_probas = model.predict_proba(random_X)
                random_data['prob'] = random_probas[:,1]
                random_pairs = random_data
                ranks = self.calculate_rank(true_pairs, random_pairs)
                mrr_res.append(np.array([1/rank for rank in ranks]).mean())
                hit1_res.append(np.array([x <= 1 for x in ranks]).mean())
                hit3_res.append(np.array([x <= 3 for x in ranks]).mean())
                hit5_res.append(np.array([x <= 5 for x in ranks]).mean())

            result['mrr'] = f"{np.array(mrr_res).mean():.3f} (+/- {np.array(mrr_res).std():.3f})"
            for k in [1,3,5]:
                res = eval(f"hit{k}_res")
                result[f'hit@{k}'] = f"{np.array(res).mean():.3f} (+/- {np.array(res).std():.3f})"

        else:

            if model_type == "graphsage_svm":
                self.logger.warning(f"Considering the calculation with {model_type} model is slow, we exclude {model_type} model for the calculation with three replacement methods.")
            else:
                self.logger.info("Using three replacement methods for MRR and Hit@K calculation")
                self.logger.warning(f"The '{model_type}' model might need to take a few hours to finish the evaluation")

                # use three replacement methods: drug-rank-based replacement, disease-rank-based replacement, combined Replacement
                test_data_pos = test_data.loc[test_data['y'] == 1,:].reset_index(drop=True)
                rank_res = self.calculate_rank_all(data_pos_df=test_data_pos, all_drug_ids=self.all_drug_ids, all_disease_ids=self.all_disease_ids, entity_embeddings=entity_embeddings, all_tp_pairs_dict=self.all_tp_pairs_dict, fitModel=model)

                ##### filtering other tp pairs #####
                drug_mrr = np.array([1/rank_res[i][0] for i in rank_res]).mean()
                result['drug_mrr'] = f"{drug_mrr:.8f}"
                disease_mrr = np.array([1/rank_res[i][1] for i in rank_res]).mean()
                result['disease_mrr'] = f"{disease_mrr:.8f}"
                both_mrr = np.array([1/rank_res[i][2] for i in rank_res]).mean()
                result['both_mrr'] = f"{both_mrr:.8f}"

                for k in [1,3,5]:
                    drug_hitk = np.array([True if rank_res[i][0] <= k else False for i in rank_res]).astype(float).mean()
                    result[f'drug_hit@{k}'] = f"{drug_hitk:.8f}"
                    disease_hitk = np.array([True if rank_res[i][1] <= k else False for i in rank_res]).astype(float).mean()
                    result[f'disease_hit@{k}'] = f"{disease_hitk:.8f}"
                    both_hitk = np.array([True if rank_res[i][2] <= k else False for i in rank_res]).astype(float).mean()
                    result[f'both_hit@{k}'] = f"{both_hitk:.8f}"

        return result