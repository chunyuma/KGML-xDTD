import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import InMemoryDataset, Data, NeighborSampler
import pickle
import math
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
import sklearn.model_selection as ms
from glob import glob
import utils
import time
import random

class ProcessedDataset(InMemoryDataset):
    def __init__(self, root, args, transform=None, pre_transform=None):
        self.args = args
        self.worker = 8

        super(ProcessedDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['processed_kg.dataset', 'train_val_test_random.pkl']

    def download(self):
        pass
    
    # @staticmethod
    # def _encode_onehot(labels):
    #     ulabels = set(labels)
    #     ulabels_dict = {c: list(np.identity(len(ulabels))[i, :]) for i, c in enumerate(ulabels)}
    #     return (np.array(list(map(ulabels_dict.get, labels)), dtype=np.int32), ulabels_dict)

    @staticmethod
    def _generate_init_emb(args, all_nodes, dim=100, known_int_emb_dict=None):

        if known_int_emb_dict is not None:
            init_embs = torch.tensor(np.vstack(list(map(known_int_emb_dict.get,all_nodes))), dtype=torch.float32)
        else:
            init_embs = torch.randn(len(all_nodes),dim)
            nn.init.xavier_normal_(init_embs)
        args.logger.info(f"Size of initial embeddings: {init_embs.shape}")
        return init_embs
    
    # @staticmethod
    # def _split_data(tp, tn, shuffle=True, batch_size=512):
    #     tp['y'] = 1
    #     tn['y'] = 0
    #     tp_num = math.ceil((tp.shape[0]/(tp.shape[0]+tn.shape[0]))*batch_size)
    #     tn_num = math.floor((tn.shape[0]/(tp.shape[0]+tn.shape[0]))*batch_size)
    #     if shuffle==True:
    #         tp = tp.sample(frac = 1)
    #         tn = tn.sample(frac = 1)
    #     tp_batch = [list(tp.index)[x:x+tp_num] for x in range(0, len(tp.index), tp_num)]
    #     tn_batch = [list(tn.index)[x:x+tn_num] for x in range(0, len(tn.index), tn_num)]
    #     if len(tp_batch) == len(tn_batch):
    #         pass
    #     elif len(tp_batch) > len(tn_batch):
    #         tn_batch += [[]]
    #     else:
    #         tp_batch += [[]]
    #     batch = [pd.concat([tp.loc[tp_batch[i],],tn.loc[tn_batch[i],]],axis=0).sample(frac=1).reset_index().drop(columns=['index']) for i in range(len(tp_batch))]
    #     return batch


    # @staticmethod
    # def _rand_rate(n, test_pairs, disease_list, idx_map, all_known_tp_pairs):

    #     random.seed(int(time.time()/100))
    #     idtoname = {value:key for key, value in idx_map.items()}
    #     ## only use the tp data
    #     test_pairs = test_pairs.loc[test_pairs['y'] == 1,:].reset_index(drop=True)
    #     drug_in_test_data = list(set(test_pairs['source']))
    #     disease_name_list = list(map(idtoname.get, disease_list))
        
    #     ## create a check list for all tp an tn pairs
    #     check_list_temp = {(all_known_tp_pairs.loc[index,'source'],all_known_tp_pairs.loc[index,'target']):1 for index in range(all_known_tp_pairs.shape[0])}
        
    #     random_pairs = []
    #     for drug in drug_in_test_data:
    #         count = 0
    #         temp_dict = dict()
    #         random.shuffle(disease_name_list)
    #         for disease in disease_name_list:
    #             if (drug, disease) not in check_list_temp and (drug, disease) not in temp_dict:
    #                 temp_dict[(drug, disease)] = 1
    #                 count += 1
    #             if count == n:
    #                 break
    #         random_pairs += [pd.DataFrame(temp_dict.keys())]
        
    #     random_pairs = pd.concat(random_pairs).reset_index(drop=True).rename(columns={0:'source',1:'target'})
    #     random_pairs['y'] = 2
        
    #     print(f'Number of random pairs: {random_pairs.shape[0]}', flush=True)

    #     return random_pairs
    
    
    def process(self):
        ## load graph data
        raw_edges = pd.read_csv(os.path.join(self.args.data_path,'graph_edges.txt'), sep='\t', header=0)
        raw_edges = raw_edges[['source','target']].drop_duplicates().reset_index(drop=True)
        all_nodes = set()
        all_nodes.update(set(raw_edges.source))
        all_nodes.update(set(raw_edges.target))
        all_nodes = list(all_nodes)

        ## load mapping files
        entity2id, _ = utils.load_index(os.path.join(self.args.data_path, 'entity2freq.txt'))
        edges = np.array(list(map(entity2id.get, np.array(raw_edges).flatten())), dtype=np.int32).reshape(np.array(raw_edges).shape) - 1
        
        ## generate initial embedding vectors for each category
        init_embs = self._generate_init_emb(self.args, all_nodes, dim=self.args.init_emb_size, known_int_emb_dict=self.args.known_int_emb_dict)
        ## generate edge index matrix
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        data = Data(feat=init_embs, edge_index=edge_index)
        train_pairs = pd.read_csv(os.path.join(self.args.data_path, 'pretrain_reward_shaping_model_train_val_test_random_data_2class', 'train_pairs.txt'), sep='\t', header=0)
        val_pairs = pd.read_csv(os.path.join(self.args.data_path, 'pretrain_reward_shaping_model_train_val_test_random_data_2class', 'val_pairs.txt'), sep='\t', header=0)
        test_pairs = pd.read_csv(os.path.join(self.args.data_path, 'pretrain_reward_shaping_model_train_val_test_random_data_2class', 'test_pairs.txt'), sep='\t', header=0)
        random_pairs = pd.read_csv(os.path.join(self.args.data_path, 'pretrain_reward_shaping_model_train_val_test_random_data_2class', 'random_pairs.txt'), sep='\t', header=0)


        # seed random state from time
        random_state2 = np.random.RandomState(int(time.time()))
        ## split training set according to the given batch size
        N = train_pairs.shape[0]//self.args.batch_size
        # Sets up 10-fold cross validation set
        cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)
        train_batch = []
        try:
            os.mkdir(os.path.join(self.processed_dir, 'train_loaders'))
        except FileExistsError:
            pass
        for _, index in cv2.split(np.array(list(train_pairs.index)), np.array(train_pairs['y'])):
            train_batch += [train_pairs.loc[list(index),:].reset_index(drop=True)]

        self.args.logger.info("generating batches for train set")
        for i in trange(len(train_batch)):

            batch_data = train_batch[i]
            data_set = set()
            data_set.update(set(batch_data.source))
            data_set.update(set(batch_data.target))
            data_idx = torch.tensor(list(map(entity2id.get, data_set)), dtype=torch.int32) - 1
            for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.args.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
                adjs = [(adj.edge_index,adj.size) for adj in adjs]
                loader = (n_id, adjs)
            filename = 'train_loader' + '_' + str(i) + '.pkl'
            with open(os.path.join(self.processed_dir, 'train_loaders', filename), 'wb') as output:
                pickle.dump(loader, output)
        
        
        ## split validation set according to the given batch size
        N = val_pairs.shape[0]//self.args.batch_size
        # Sets up 10-fold cross validation set
        cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)
        val_batch = []
        try:
            os.mkdir(os.path.join(self.processed_dir, 'val_loaders'))
        except FileExistsError:
            pass
        for _, index in cv2.split(np.array(list(val_pairs.index)), np.array(val_pairs['y'])):
            val_batch += [val_pairs.loc[list(index),:].reset_index(drop=True)]

        self.args.logger.info("generating batches for validation set")
        for i in trange(len(val_batch)):

            batch_data = val_batch[i]
            data_set = set()
            data_set.update(set(batch_data.source))
            data_set.update(set(batch_data.target))
            data_idx= torch.tensor(list(map(entity2id.get, data_set)), dtype=torch.int32) - 1
            for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.args.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
                adjs = [(adj.edge_index,adj.size) for adj in adjs]
                loader = (n_id, adjs)
            filename = 'val_loader' + '_' + str(i) + '.pkl'
            with open(os.path.join(self.processed_dir, 'val_loaders', filename), 'wb') as output:
                pickle.dump(loader, output)            
            
        
        ## split test set according to the given batch size
        N = test_pairs.shape[0]//self.args.batch_size
        # Sets up 10-fold cross validation set
        cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)    
        test_batch = []
        try:
            os.mkdir(os.path.join(self.processed_dir, 'test_loaders'))
        except FileExistsError:
            pass
        for _, index in cv2.split(np.array(list(test_pairs.index)), np.array(test_pairs['y'])):
            test_batch += [test_pairs.loc[list(index),:].reset_index(drop=True)]

        self.args.logger.info("generating batches for test set")
        for i in trange(len(test_batch)):

            batch_data = test_batch[i]
            data_set = set()
            data_set.update(set(batch_data.source))
            data_set.update(set(batch_data.target))
            data_idx= torch.tensor(list(map(entity2id.get, data_set)), dtype=torch.int32) - 1
            for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.args.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
                adjs = [(adj.edge_index,adj.size) for adj in adjs]
                loader = (n_id, adjs)
            filename = 'test_loader' + '_' + str(i) + '.pkl'
            with open(os.path.join(self.processed_dir, 'test_loaders', filename), 'wb') as output:
                pickle.dump(loader, output)

                
        ## split random pair set according to the given batch size
        N = random_pairs.shape[0]//(self.args.batch_size * 3)
        # Sets up 10-fold cross validation set
        cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)    
        random_batch = []
        try:
            os.mkdir(os.path.join(self.processed_dir, 'random_loaders'))
        except FileExistsError:
            pass
        for _, index in cv2.split(np.array(list(random_pairs.index)), np.array(random_pairs['y'])):
            random_batch += [random_pairs.loc[list(index),:].reset_index(drop=True)]

        self.args.logger.info("generating batches for random pair set")
        for i in trange(len(random_batch)):

            batch_data = random_batch[i]
            data_set = set()
            data_set.update(set(batch_data.source))
            data_set.update(set(batch_data.target))
            data_idx=torch.tensor(list(map(entity2id.get, data_set)), dtype=torch.int32) - 1
            for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.args.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
                adjs = [(adj.edge_index,adj.size) for adj in adjs]
                loader = (n_id, adjs)
            filename = 'random_loader' + '_' + str(i) + '.pkl'
            with open(os.path.join(self.processed_dir, 'random_loaders', filename), 'wb') as output:
                pickle.dump(loader, output)

        train_val_test_random = [train_batch, val_batch, test_batch, random_batch]
        
        with open(os.path.join(self.processed_dir, 'train_val_test_random.pkl'), 'wb') as output:
            pickle.dump(train_val_test_random, output)

        torch.save(data, os.path.join(self.processed_dir, 'processed_kg.dataset'))
        
    def get_dataset(self):
        data = torch.load(os.path.join(self.processed_dir, 'processed_kg.dataset'))
        return data
    
    def get_train_val_test_random(self):
        train_val_test_random = pickle.load(open(os.path.join(self.processed_dir, 'train_val_test_random.pkl'), 'rb'))
        return train_val_test_random

    def get_train_loader(self):
        train_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'train_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return DataLoader(utils.DataWrapper(train_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)

    def get_val_loader(self):
        val_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'val_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return DataLoader(utils.DataWrapper(val_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)

    def get_test_loader(self):
        test_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'test_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return DataLoader(utils.DataWrapper(test_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)

    def get_random_loader(self):
        random_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'random_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return DataLoader(utils.DataWrapper(random_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)

