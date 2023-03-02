import sys, os
from typing import List
import eval_utilities
import pickle
import numpy as np
import pandas as pd
import graph_tool.all as gt
import itertools
import torch
from tqdm import tqdm

class KGML_xDTD:

    def __init__(self, args, data_path: str, model_path: str):
        """
        Initialize KGML_xDTD
        :param args: argparse.ArgumentParser.parse_args object
        :param data_path: path to the data directory
        :param model_path: path to the model directory
        """
        ## set up args
        self.args = eval_utilities.set_args(args)
        self.args.data_dir = data_path
        self.args.logger = eval_utilities.get_logger()
        if self.args.use_gpu:
            torch.cuda.set_device(self.args.gpu)

        ## check device
        self.args.use_gpu, self.args.device = eval_utilities.check_device(logger = self.args.logger, use_gpu = self.args.use_gpu, gpu = self.args.gpu)
        self.args.logger.info(f"Use device {self.args.device}")

        ## load datasets
        self.args.logger.info(f"Loading datasets")
        self.entity_embeddings_dict = eval_utilities.load_graphsage_unsupervised_embeddings(self.args.data_dir)
        self.args.entity2id, self.args.id2entity = eval_utilities.load_index(os.path.join(self.args.data_dir, 'entity2freq.txt'))
        self.args.relation2id, self.args.id2relation = eval_utilities.load_index(os.path.join(self.args.data_dir, 'relation2freq.txt'))
        self.args.type2id, self.args.id2type = eval_utilities.load_index(os.path.join(self.args.data_dir, 'type2freq.txt'))
        with open(os.path.join(self.args.data_dir, 'entity2typeid.pkl'), 'rb') as infile:
            self.args.entity2typeid = pickle.load(infile)
        drug_type = ['biolink:Drug', 'biolink:SmallMolecule']
        drug_type_ids = [self.args.type2id[x] for x in drug_type]
        self.drug_curie_ids = [self.args.id2entity[index] for index, typeid in enumerate(self.args.entity2typeid) if typeid in drug_type_ids]
        self.drug_curie_names = [value['preferred_name'] if value else None for _, value in eval_utilities.get_preferred_curie(curie_list=self.drug_curie_ids).items()]
        disease_type = ['biolink:Disease', 'biolink:PhenotypicFeature', 'biolink:BehavioralFeature','biolink:DiseaseOrPhenotypicFeature']
        disease_type_ids = [self.args.type2id[x] for x in disease_type]
        self.disease_curie_ids = [self.args.id2entity[index] for index, typeid in enumerate(self.args.entity2typeid) if typeid in disease_type_ids]
        self.disease_curie_names = [value['preferred_name'] if value else None for _, value in eval_utilities.get_preferred_curie(curie_list=self.disease_curie_ids).items()]

        ## load drug repurposing module
        self.args.logger.info(f"Loading drug repurposing module")
        self.drp_module = eval_utilities.load_drp_module(model_path)

        ## load mechanism of action module
        self.args.logger.info(f"Loading mechanism of action module")
        self.kg, self.env, self.moa_module = eval_utilities.load_moa_module(args, model_path)

        ## load graph
        self.G, self.etype = eval_utilities.load_gt_kg(self.kg)

        ## other variables
        self._prob_table_cache = dict()
        self._topN_prob_drug_cache = dict()
        self._topN_prob_disease_cache = dict()
        self._filtered_res_all_paths = dict()
        self._topM_paths = dict()

    def _get_preferred_curie(self, curie: str = None, name: str = None, curie_type: str = 'drug'):
        """
        Get the preferred curie based on a given curie or its name
        :param curie: curie id [optional]. Note one of `curie` or `name` is required.
        :param name: name of curie [optional]. Note one of `curie` or `name` is required.
        :param curie_type: is drug or disease [optional]. Default is drug
        :return: a treatment probability
        :rtype: float
        """ 
        if curie_type not in ['drug', 'disease']:
            self.args.logger.error(f"The parameter 'curie_type' should be either 'drug' or 'disease'.")
            return None

        if curie_type == 'drug': 
            curie_ids = self.drug_curie_ids
        else:
            curie_ids = self.disease_curie_ids

        if curie:
            self.args.logger.info(f"Get {curie_type} curie {curie}.")
            normalized_curie = eval_utilities.get_preferred_curie(curie=curie)
            if normalized_curie:
                preferred_curie = normalized_curie['preferred_curie']
                if not (preferred_curie in curie_ids):
                    self.args.logger.error(f"Likely the model was not trained with the given {curie_type} curie {curie}.")
                    return None
                else:
                    self.args.logger.info(f"Switch given {curie_type} curie to the preferred curie {preferred_curie}")
            else:
                self.args.logger.error(f"Could not get preferred curie for the given {curie_type} curie {curie}. Likely the model was not trained with this {curie_type}.")
                return None
        elif name:
            self.args.logger.info(f"Get {curie_type} name {name}.")
            normalized_curie = eval_utilities.get_preferred_curie(name=name)
            if normalized_curie:
                preferred_curie = normalized_curie['preferred_curie']
                if not (preferred_curie in curie_ids):
                    self.args.logger.error(f"Likely the model was not trained with the given {curie_type} name {name}.")
                    return None
                else:
                    self.args.logger.info(f"Switch given {curie_type} name to the preferred curie {preferred_curie}")
            else:
                self.args.logger.error(f"Could not get preferred curie for the given {curie_type} name {name}. Likely the model was not trained with this {curie_type}.")
                return None       
        else:
            self.args.logger.error(f"No '{curie_type}_curie' or '{curie_type}_name' provided. Please provide either of them.")
            return None

        return preferred_curie


    def predict_single_ddp(self, drug_curie: str = None, drug_name: str = None, disease_curie: str = None, disease_name: str = None):
        """
        Predict a treatment probability for a single drug-diseae pair
        :param drug_curie: curie id of a drug [optional]. Note one of `drug_curie` or `drug_name` is required.
        :param drug_name: name of a drug [optional]. Note one of `drug_curie` or `drug_name` is required.
        :param disease_curie: curie id of a disease [optional]. Note one of `disease_curie` or `disease_name` is required.
        :param disease_name: name of a disease [optional]. Note one of `disease_curie` or `disease_name` is required.
        :return: a treatment probability
        :rtype: float
        """

        ## get preferred_curie for a given drug curie or name
        if drug_curie:
            orig_drug = drug_curie
            preferred_drug_curie = self._get_preferred_curie(curie=drug_curie, curie_type= 'drug')
        elif drug_name:
            orig_drug = drug_name
            preferred_drug_curie = self._get_preferred_curie(name=drug_name, curie_type= 'drug')
        else:
            self.args.logger.error(f"No 'drug_curie' or 'drug_name' provided. Please provide either of them.")
            return None
        if not preferred_drug_curie:
            return None

        ## get preferred_curie for a given disease curie or name
        if disease_curie:
            orig_disease = disease_curie
            preferred_disease_curie = self._get_preferred_curie(curie=disease_curie, curie_type= 'disease')
        elif disease_name:
            orig_disease = disease_name
            preferred_disease_curie = self._get_preferred_curie(name=disease_name, curie_type= 'disease')
        else:
            self.args.logger.error(f"No 'disease_curie' or 'disease_name' provided. Please provide either of them.")
            return None
        if not preferred_disease_curie:
            return None

        self.args.logger.info(f"Predicting the treatment probability for a drug-disease pair {(orig_drug,orig_disease)}")
        if (preferred_drug_curie,preferred_disease_curie) in self._prob_table_cache:
            res_temp = self._prob_table_cache[(preferred_drug_curie,preferred_disease_curie)]
            return res_temp[1]
        else:
            X = np.vstack([np.hstack([self.entity_embeddings_dict[preferred_drug_curie],self.entity_embeddings_dict[preferred_disease_curie]])])
            res_temp = self.drp_module.predict_proba(X)[0]
            self._prob_table_cache[(preferred_drug_curie,preferred_disease_curie)] = res_temp
            return res_temp[1]


    def predict_ddps(self, drug_disease_curie_list: List = None, drug_disease_name_list: List = None):
        """
        Predict treatment probabilities for a list of drug-diseae pairs
        :param drug_disease_curie_list: a list of drug-disease pairs [optional]. Note one of `drug_disease_curie_list` or `drug_disease_name_list` is required.
        :param drug_disease_name_list: a list of drug-disease pairs [optional]. Note one of `drug_disease_curie_list` or `drug_disease_name_list` is required.
        :return: a list of filtered drug-diseae pairs with their corresponding predicted probabilities
        :rtype: tuple[List, numpy.ndarray]
        """

        if drug_disease_curie_list:
            drug_curie_list, disease_curie_list = list(zip(*drug_disease_curie_list))
            dict_normalized_drug = eval_utilities.get_preferred_curie(curie_list=list(drug_curie_list))
            normalized_drug_curie_list = [dict_normalized_drug[curie]['preferred_curie'] if dict_normalized_drug[curie] else None for curie in drug_curie_list]
            dict_normalized_disease = eval_utilities.get_preferred_curie(curie_list=list(disease_curie_list))
            normalized_disease_curie_list = [dict_normalized_disease[curie]['preferred_curie'] if dict_normalized_disease[curie] else None for curie in disease_curie_list]
            temp_df = pd.DataFrame({'drug_curie':drug_curie_list,'disease_curie':disease_curie_list,'normalized_drug_curie':normalized_drug_curie_list,'normalized_disease_curie':normalized_disease_curie_list})
            temp_df['drug_exist'] = temp_df['normalized_drug_curie'].isin(self.drug_curie_ids)
            temp_bad_list = list(set(temp_df['drug_curie'][~temp_df['drug_exist']].tolist()))
            if len(temp_bad_list) > 0:
                self.args.logger.warning(f"the model was not trained with the following drugs: {temp_bad_list}.")
            temp_df['disease_exist'] = temp_df['normalized_disease_curie'].isin(self.disease_curie_ids)
            temp_bad_list = list(set(temp_df['disease_curie'][~temp_df['disease_exist']].tolist()))
            if len(temp_bad_list) > 0:
                self.args.logger.warning(f"the model was not trained with the following drugs: {temp_bad_list}.")
            temp_df['prob'] = -1 ## set default value
            orig_drug_disease_list = temp_df.loc[temp_df['drug_exist'] & temp_df['disease_exist'],('drug_curie','disease_curie')].values
            bad_drug_disease_list = temp_df.loc[~(temp_df['drug_exist'] & temp_df['disease_exist']),('drug_curie','disease_curie')].values
            drug_disease_list = temp_df.loc[temp_df['drug_exist'] & temp_df['disease_exist'],('normalized_drug_curie','normalized_disease_curie')].values
        elif drug_disease_name_list:
            drug_name_list, disease_name_list = list(zip(*drug_disease_name_list))
            dict_normalized_drug = eval_utilities.get_preferred_curie(name_list=list(drug_name_list))
            normalized_drug_curie_list = [dict_normalized_drug[name]['preferred_curie'] if dict_normalized_drug[name] else None for name in drug_name_list]
            dict_normalized_disease = eval_utilities.get_preferred_curie(name_list=list(disease_name_list))
            normalized_disease_curie_list = [dict_normalized_disease[name]['preferred_curie'] if dict_normalized_disease[name] else None for name in disease_name_list]
            temp_df = pd.DataFrame({'drug_name':drug_name_list,'disease_name':disease_name_list,'normalized_drug_curie':normalized_drug_curie_list,'normalized_disease_curie':normalized_disease_curie_list})
            temp_df['drug_exist'] = temp_df['normalized_drug_curie'].isin(self.drug_curie_ids)
            temp_bad_list = list(set(temp_df['drug_name'][~temp_df['drug_exist']].tolist()))
            if len(temp_bad_list) > 0:
                self.args.logger.warning(f"the model was not trained with the following diseases: {temp_bad_list}.")
            temp_df['disease_exist'] = temp_df['normalized_disease_curie'].isin(self.disease_curie_ids)
            temp_bad_list = list(set(temp_df['disease_name'][~temp_df['disease_exist']].tolist()))
            if len(temp_bad_list) > 0:
                self.args.logger.warning(f"the model was not trained with the following diseases: {temp_bad_list}.")
            temp_df['prob'] = -1 ## set default value
            orig_drug_disease_list = temp_df.loc[temp_df['drug_exist'] & temp_df['disease_exist'],('drug_name','disease_name')].values
            bad_drug_disease_list = temp_df.loc[~(temp_df['drug_exist'] & temp_df['disease_exist']),('drug_name','disease_name')].values
            drug_disease_list = temp_df.loc[temp_df['drug_exist'] & temp_df['disease_exist'],('normalized_drug_curie','normalized_disease_curie')].values
        else:
            self.args.logger.error(f"No 'drug_disease_curie_list' or 'drug_disease_name_list' provided.")
            return None            

        if len(bad_drug_disease_list) != 0:
            self.args.logger.info(f"Predicting the treatment probabilities for the following drug-disease pairs after filtering the 'bad' pairs:\n {bad_drug_disease_list}")
        else:
            self.args.logger.info("Predicting the treatment probabilities for the following drug-disease pairs")
        if len(drug_disease_list) == 0:
            self.args.logger.warning("No 'good' drug-disase pairs after filtering the 'bad' pairs")
            return None

        X = np.vstack([np.hstack([self.entity_embeddings_dict[drug_curie],self.entity_embeddings_dict[disease_curie]]) for drug_curie, disease_curie in drug_disease_list])
        res_temp = self.drp_module.predict_proba(X)
        self._prob_table_cache.update({(drug_curie, disease_curie):res for (drug_curie, disease_curie), res in zip(drug_disease_list, res_temp)})
        temp_df.loc[temp_df['drug_exist'] & temp_df['disease_exist'],'prob'] = res_temp[:,1]
        return orig_drug_disease_list,  temp_df.loc[temp_df['drug_exist'] & temp_df['disease_exist'],'prob'].values


    def predict_top_N_diseases(self, drug_curie: str = None, drug_name: str = None,  N: int = 10):
        """
        Predict top N potential diseases that could be treated by a given drug
        :param drug_curie: curie id of a drug [optional]. Note one of `drug_curie` or `drug_name` is required.
        :param drug_name: name of a drug [optional]. Note one of `drug_curie` or `drug_name` is required.
        :param N: Number of top potential diseases with their treatment probabiliteis for a given drug
        :return: a dataframe with the top potential diseases (curie ids, names) with their treatment probabiliteis
        :rtype: dataframe
        """

        ## get preferred_curie for a given drug curie or name
        if drug_curie:
            orig_drug = drug_curie
            preferred_drug_curie = self._get_preferred_curie(curie=drug_curie, curie_type= 'drug')
        elif drug_name:
            orig_drug = drug_name
            preferred_drug_curie = self._get_preferred_curie(name=drug_name, curie_type= 'drug')
        else:
            self.args.logger.error(f"No 'drug_curie' or 'drug_name' provided. Please provide either of them.")
            return None
        if not preferred_drug_curie:
            return None

        if preferred_drug_curie in self._topN_prob_drug_cache:
            res = self._topN_prob_drug_cache[preferred_drug_curie]
            top_N_diseases = res.iloc[:N,:]
        else:
            self.args.logger.info(f"Predicting top{N} diseases for drug {orig_drug}")
            X = np.vstack([np.hstack([self.entity_embeddings_dict[preferred_drug_curie],self.entity_embeddings_dict[disease_curie_id]]) for disease_curie_id in self.disease_curie_ids])
            res_temp = self.drp_module.predict_proba(X)
            res = pd.concat([pd.DataFrame(self.disease_curie_ids),pd.DataFrame(self.disease_curie_names),pd.DataFrame(res_temp)], axis=1)
            res.columns = ['disease_id','disease_name','tn_score','tp_score','unknown_score']
            res = res.sort_values(by=['tp_score'],ascending=False).reset_index(drop=True)
            self._topN_prob_drug_cache[preferred_drug_curie] = res
            top_N_diseases = res.iloc[:N,:]
        
        return top_N_diseases

    def predict_top_N_drugs(self, disease_curie: str = None, disease_name: str = None, N: int = 10):
        """
        Predict top N potential drugs that could be used to treat a given disease
        :param disease_curie: curie id of a disease [optional]. Note one of `disease_curie` or `disease_name` is required.
        :param disease_name: name of a disease [optional]. Note one of `disease_curie` or `disease_name` is required.
        :param N: Number of top potential drugs with their treatment probabiliteis for a given disease
        :return: a dataframe with the top potential drugs (curie ids, names) with their treatment probabiliteis
        :rtype: dataframe
        """

        ## get preferred_curie for a given disease curie or name
        if disease_curie:
            orig_disease = disease_curie
            preferred_disease_curie = self._get_preferred_curie(curie=disease_curie, curie_type= 'disease')
        elif disease_name:
            orig_disease = disease_name
            preferred_disease_curie = self._get_preferred_curie(name=disease_name, curie_type= 'disease')
        else:
            self.args.logger.error(f"No 'disease_curie' or 'disease_name' provided. Please provide either of them.")
            return None
        if not preferred_disease_curie:
            return None

        if preferred_disease_curie in self._topN_prob_disease_cache:
            res = self._topN_prob_disease_cache[preferred_disease_curie]
            top_N_drugs = res.iloc[:N,:]
        else:
            self.args.logger.info(f"Predicting top{N} drugs for disease {orig_disease}")
            X = np.vstack([np.hstack([self.entity_embeddings_dict[drug_curie_id],self.entity_embeddings_dict[preferred_disease_curie]]) for drug_curie_id in self.drug_curie_ids])
            res_temp = self.drp_module.predict_proba(X)
            res = pd.concat([pd.DataFrame(self.drug_curie_ids),pd.DataFrame(self.drug_curie_names),pd.DataFrame(res_temp)], axis=1)
            res.columns = ['drug_id','drug_name','tn_score','tp_score','unknown_score']
            res = res.sort_values(by=['tp_score'],ascending=False).reset_index(drop=True)
            self._topN_prob_disease_cache[preferred_disease_curie] = res
            top_N_drugs = res.iloc[:N,:]
        
        return top_N_drugs

    def _extract_all_paths(self, df_table: pd.core.frame.DataFrame = None):
        """
        Extract all paths from KG based on given dataframe with the queried drug-disease pairs
        :param df_table: a dataframe with the queried drug-disease pairs [Required] 
        :return 0 or 1:
        :rtype: integer
        """        
        if df_table is None:
            self.args.logger.warning(f"No 'df_table' is given. Please provide a dataframe with the queried drug-disease pairs.")
            return 0

        if not isinstance(df_table, pd.core.frame.DataFrame):
            self.args.logger.warning(f"No 'df_table' should be a DataFrame but {type(df_table)} is given.")
            return 0

        df_talbe_list = df_table.to_numpy()
        if len(df_talbe_list) == 0:
            self.args.logger.warning("The 'df_table' is empty.")
            return 0
        elif len(df_talbe_list) == 1:
            self.args.logger.info(f"Extracting all paths with the length of up to 3 for the drug-disease pair {[(drug,disease) for drug, disease in df_talbe_list]}.")
        else:
            self.args.logger.info(f"Extracting all paths with the length of up to 3 for {len(df_table)} drug-disease pairs {df_table}.")
        filter_edges = [self.args.relation2id[edge] for edge in ['biolink:related_to','biolink:biolink:part_of','biolink:coexists_with','biolink:contraindicated_for'] if self.args.relation2id.get(edge)]
        
        for row in df_talbe_list:
            source, target = row
            if (source,target) not in self._filtered_res_all_paths:
                all_paths = [list(path) for path in gt.all_paths(self.G, self.args.entity2id[source], self.args.entity2id[target], cutoff=3)]
                entity_paths = []
                relation_paths = []
                for path in all_paths:
                    path_temp = []
                    for index in range(len(path)-1):
                        if index == 0:
                            path_temp += [path[index], list(self.etype[self.G.edge(path[index], path[index+1])]), path[index+1]]
                        else:
                            path_temp += [list(self.etype[self.G.edge(path[index], path[index+1])]), path[index+1]]
                    flattened_paths = list(itertools.product(*map(lambda x: [x] if type(x) is not list else x, path_temp)))
                    for flattened_path in flattened_paths:
                        if len(flattened_path) == 7:
                            relation_paths += [[self.args.relation2id['SELF_LOOP_RELATION']] + [x for index3, x in enumerate(flattened_path) if index3%2==1]]
                            entity_paths += [[x for index3, x in enumerate(flattened_path) if index3%2==0]]
                        elif len(flattened_path) == 5:
                            relation_paths += [[self.args.relation2id['SELF_LOOP_RELATION']] + [x for index3, x in enumerate(flattened_path) if index3%2==1] + [self.args.relation2id['SELF_LOOP_RELATION']]]
                            entity_paths += [[x for index3, x in enumerate(flattened_path) if index3%2==0] + [flattened_path[-1]]]
                        else:
                            self.args.logger.info(f"Found weird path: {flattened_path}")
                edge_mat = torch.tensor(relation_paths)
                node_mat = torch.tensor(np.array(entity_paths).astype(int))
                temp = pd.DataFrame(edge_mat.numpy())
                if len(temp) != 0:
                    keep_index = list(temp.loc[~(temp[1].isin(filter_edges) | temp[2].isin(filter_edges) | temp[3].isin(filter_edges)),:].index)
                    if len(keep_index) == 0:
                        self._filtered_res_all_paths[(source,target)] = []
                    else:
                        self._filtered_res_all_paths[(source,target)] = [edge_mat[keep_index],node_mat[keep_index]]
                else:
                    self._filtered_res_all_paths[(source,target)] = []
        return 1


    def _make_path(self, rel_ent_score):
        rel_vec, ent_vec, score = rel_ent_score
        return ('->'.join([eval_utilities.id_to_name(self.args.id2entity[ent_vec[index]])+'->'+self.args.id2relation[rel_vec[index+1]] for index in range(len(ent_vec)-1)] + [eval_utilities.id_to_name(self.args.id2entity[ent_vec[len(ent_vec)-1]])]), score)

    def _batch_get_true(self, args, batch_action_spaces, batch_true_actions):
        ((batch_r_space, batch_e_space), batch_action_mask) = batch_action_spaces
        if args.use_gpu:
            true_r = batch_true_actions[0].view(-1,1).cuda()
        else:
            true_r = batch_true_actions[0].view(-1,1)
        if args.use_gpu:
            true_e = batch_true_actions[1].view(-1,1).cuda()
        else:
            true_e = batch_true_actions[1].view(-1,1)
        true_idx_in_actor = torch.where((batch_r_space == true_r) * (batch_e_space == true_e))[1]

        return true_idx_in_actor, (true_r, true_e)

    def _select_true_action(self, batch_state, batch_action_spaces, batch_true_actions):

        device = self.args.device
        state_inputs = self.moa_module.process_state(self.moa_module.history_len, batch_state).to(device)
        true_idx_in_actor, true_next_actions = self._batch_get_true(self.args, batch_action_spaces, batch_true_actions)

        probs, _ = self.moa_module.policy_net(state_inputs, batch_action_spaces)
        if self.args.use_gpu:
            eval_utilities.empty_gpu_cache(self.args)
        true_idx_in_actor = true_idx_in_actor.to(device)
        true_prob = probs.gather(1, true_idx_in_actor.view(-1, 1)).view(-1)
        weighted_logprob = torch.log((true_prob.view(-1,1)+eval_utilities.TINY_VALUE) * torch.count_nonzero(probs, dim=1).view(-1,1))

        return true_next_actions, weighted_logprob

    def _batch_calculate_prob_score(self, batch_paths: List[np.ndarray]):
        """
        Calculate the scores of KG-based paths in a batch based on the trained MOA module
        :param batch_paths: a batch of KG-based path [required]
        :return: an array of path scores
        :rtype: np.ndarray
        """

        self.env.reset()
        self.moa_module.policy_net.eval()
        dataloader = eval_utilities.ACDataLoader(list(range(batch_paths[1].shape[0])), self.args.batch_size, permutation=False)

        # pbar = tqdm(total=dataloader.num_paths)
        pred_prob_scores = []
        while dataloader.has_next():

            batch_path_id = dataloader.get_batch()
            source_ids = batch_paths[1][batch_path_id][:,0]
            self.env.initialize_path(source_ids)
            act_num = 1

            if self.args.use_gpu:
                action_log_weighted_prob = eval_utilities.zeros_var_cuda(len(batch_path_id), args=self.args, use_gpu=True)
            else:
                action_log_weighted_prob = eval_utilities.zeros_var_cuda(len(batch_path_id), args=self.args, use_gpu=False)

            while not self.env._done:
                batch_true_action = [batch_paths[0][batch_path_id][:,act_num], batch_paths[1][batch_path_id][:,act_num]]
                true_next_act, weighted_logprob = self._select_true_action(self.env._batch_curr_state, self.env._batch_curr_action_spaces, batch_true_action)
                self.env.batch_step(true_next_act)
                if self.args.use_gpu:
                    eval_utilities.empty_gpu_cache(self.args)
                action_log_weighted_prob = action_log_weighted_prob.view(-1, 1) + self.args.factor**(act_num-1) * weighted_logprob
                if self.args.use_gpu:
                    eval_utilities.empty_gpu_cache(self.args)
                act_num += 1
            ### End of episodes ##

            pred_prob_scores += [action_log_weighted_prob.view(-1).cpu().detach()]
            self.env.reset()

            if self.args.use_gpu:
                eval_utilities.empty_gpu_cache(self.args)
            # pbar.update(len(source_ids))

        return np.concatenate(pred_prob_scores)


    def predict_top_M_moa_paths(self, drug_curie: str = None, drug_name: str = None, disease_curie: str = None, disease_name: str = None, M: int = 10):
        """
        Predict top M potential KG-based MOA paths for explaining the treatment relationship of a single drug-diseae pair
        :param drug_curie: curie id of a drug [optional]. Note one of `drug_curie` or `drug_name` is required.
        :param drug_name: name of a drug [optional]. Note one of `drug_curie` or `drug_name` is required.
        :param disease_curie: curie id of a disease [optional]. Note one of `disease_curie` or `disease_name` is required.
        :param disease_name: name of a disease [optional]. Note one of `disease_curie` or `disease_name` is required.
        :param M: Number of predicted KG-based MOA paths to return
        :return: a list of paths with their corresponding scores
        :rtype: List[tuple]
        """

        ## get preferred_curie for a given drug curie or name
        if drug_curie:
            original_drug = drug_curie
            preferred_drug_curie = self._get_preferred_curie(curie=drug_curie, curie_type= 'drug')
        elif drug_name:
            original_drug = drug_name
            preferred_drug_curie = self._get_preferred_curie(name=drug_name, curie_type= 'drug')
        else:
            self.args.logger.error(f"No 'drug_curie' or 'drug_name' provided. Please provide either of them.")
            return None
        if not preferred_drug_curie:
            return None

        ## get preferred_curie for a given disease curie or name
        if disease_curie:
            original_disease = disease_curie
            preferred_disease_curie = self._get_preferred_curie(curie=disease_curie, curie_type= 'disease')
        elif disease_name:
            original_disease = disease_name
            preferred_disease_curie = self._get_preferred_curie(name=disease_name, curie_type= 'disease')
        else:
            self.args.logger.error(f"No 'disease_curie' or 'disease_name' provided. Please provide either of them.")
            return None
        if not preferred_disease_curie:
            return None

        self.args.logger.info(f"Predicting top M potential KG-based MOA paths for a drug-disease pair {(original_drug,original_disease)}")
        if (preferred_drug_curie, preferred_disease_curie) in self._filtered_res_all_paths:
            if (preferred_drug_curie, preferred_disease_curie) in self._topM_paths:
                if len(self._topM_paths[(preferred_drug_curie, preferred_disease_curie)]) == 0:
                    self.args.logger.warning(f"No path with length up to 3 exists between {original_drug} and {original_disease} in KG")
                    return []
                else:
                    self.args.logger.info(f"Calculating all paths' scores")
                    batch_paths_sorted = self._topM_paths[(preferred_drug_curie, preferred_disease_curie)]
                    ## select the top M paths (Note that the paths with the same nodes are counted as the same path)
                    temp_dict = dict()
                    count = 0
                    top_indexes = []
                    for index, x in enumerate(batch_paths_sorted[1].numpy()):
                        if tuple(x) in temp_dict:
                            top_indexes += [index]
                        else:
                            count += 1
                            temp_dict[tuple(x)] = 1
                            top_indexes += [index]
                        if count == M:
                            break
                    res = [batch_paths_sorted[0][top_indexes], batch_paths_sorted[1][top_indexes], batch_paths_sorted[2][top_indexes]]
                    return [self._make_path([res[0][index].numpy(),res[1][index].numpy(), res[2][index].numpy().item()]) for index in range(len(res[0]))]

        else:
            df_table = pd.DataFrame({'drug':[preferred_drug_curie],'disease':[preferred_disease_curie]})
            setup = self._extract_all_paths(df_table)
            if setup:
                if len(self._filtered_res_all_paths[(preferred_drug_curie, preferred_disease_curie)]) == 0:
                    self.args.logger.warning(f"No path with length up to 3 exists between {original_drug} and {original_disease} in KG")
                    return []
                else:
                    ## set up some filtering rules
                    filter_edges = [self.args.relation2id[edge] for edge in ['biolink:related_to','biolink:biolink:part_of','biolink:coexists_with','biolink:contraindicated_for'] if self.args.relation2id.get(edge)]
                    batch_paths = self._filtered_res_all_paths[(preferred_drug_curie, preferred_disease_curie)]
                    ## filter out some paths with unwanted predicates for explanations
                    temp = pd.DataFrame(batch_paths[0].numpy())
                    keep_index = list(temp.loc[~(temp[1].isin(filter_edges) | temp[2].isin(filter_edges) | temp[3].isin(filter_edges)),:].index)
                    batch_paths = [batch_paths[0][keep_index],batch_paths[1][keep_index]]
                    ## calculate path prob score
                    pred_prob_scores = self._batch_calculate_prob_score(batch_paths)
                    pred_prob_scores = torch.tensor(pred_prob_scores)
                    sorted_scores, indices = torch.sort(pred_prob_scores, descending=True)
                    batch_paths_sorted = [batch_paths[0][indices], batch_paths[1][indices], sorted_scores]
                    self._topM_paths[(preferred_drug_curie, preferred_disease_curie)] = batch_paths_sorted
                    ## select the top M paths (Note that the paths with the same nodes are counted as the same path)
                    temp_dict = dict()
                    count = 0
                    top_indexes = []
                    for index, x in enumerate(batch_paths_sorted[1].numpy()):
                        if tuple(x) in temp_dict:
                            top_indexes += [index]
                        else:
                            count += 1
                            temp_dict[tuple(x)] = 1
                            top_indexes += [index]
                        if count == M:
                            break
                    res = [batch_paths_sorted[0][top_indexes], batch_paths_sorted[1][top_indexes], batch_paths_sorted[2][top_indexes]]
                    return [self._make_path([res[0][index].numpy(),res[1][index].numpy(), res[2][index].numpy().item()]) for index in range(len(res[0]))]


