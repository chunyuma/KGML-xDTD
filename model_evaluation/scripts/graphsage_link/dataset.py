import pandas as pd
import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import InMemoryDataset, Data, NeighborSampler
import pickle
from tqdm import tqdm, trange
import sklearn.model_selection as ms
from glob import glob
import time

pathlist = os.getcwd().split(os.path.sep)
Rootindex = pathlist.index("KGML-xDTD")
sys.path.append(os.path.sep.join([*pathlist[:(Rootindex + 1)], 'model_evaluation', 'scripts']))
import eval_utilities


class Dataset(InMemoryDataset):
    def __init__(self, data_path, logger, known_int_emb_dict = None, transform=None, pre_transform=None):
        self.known_int_emb_dict = known_int_emb_dict
        self.init_emb_size = 100
        self.batch_size = 512
        self.layer_size = [96, 96]
        self.data_path = data_path
        self.worker = 16
        self.logger = logger
        root = os.path.join(data_path, 'graphsage_link', 'ProcessedDataset')
        if not os.path.exists(root):
            os.mkdirs(root)

        super(Dataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['processed_kg.dataset', 'train_val_test_random.pkl']

    def download(self):
        pass

    def _generate_init_emb(self, all_nodes, dim=100, known_int_emb_dict=None):

        if known_int_emb_dict is not None:
            init_embs = torch.tensor(np.vstack(list(map(known_int_emb_dict.get,all_nodes))), dtype=torch.float32)
        else:
            init_embs = torch.randn(len(all_nodes),dim)
            nn.init.xavier_normal_(init_embs)
        self.logger.info(f"Size of initial embeddings: {init_embs.shape}")
        return init_embs
    
    def process(self):
        ## load graph data
        raw_edges = pd.read_csv(os.path.join(self.data_path,'graph_edges.txt'), sep='\t', header=0)
        raw_edges = raw_edges[['source','target']].drop_duplicates().reset_index(drop=True)
        all_nodes = set()
        all_nodes.update(set(raw_edges.source))
        all_nodes.update(set(raw_edges.target))
        all_nodes = list(all_nodes)

        ## load mapping files
        entity2id, _ = eval_utilities.load_index(os.path.join(self.data_path, 'entity2freq.txt'))
        edges = np.array(list(map(entity2id.get, np.array(raw_edges).flatten())), dtype=np.int32).reshape(np.array(raw_edges).shape) - 1
        
        ## generate initial embedding vectors for each category
        init_embs = self._generate_init_emb(all_nodes, dim=self.init_emb_size, known_int_emb_dict=self.known_int_emb_dict)
        ## generate edge index matrix
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        data = Data(feat=init_embs, edge_index=edge_index)
        train_pairs = pd.read_csv(os.path.join(self.data_path, 'train_pairs.txt'), sep='\t', header=0)
        train_pairs = train_pairs.loc[train_pairs['y']!=2,:].reset_index(drop=True)
        val_pairs = pd.read_csv(os.path.join(self.data_path, 'val_pairs.txt'), sep='\t', header=0)
        val_pairs = val_pairs.loc[val_pairs['y']!=2,:].reset_index(drop=True)
        test_pairs = pd.read_csv(os.path.join(self.data_path, 'test_pairs.txt'), sep='\t', header=0)
        test_pairs = test_pairs.loc[test_pairs['y']!=2,:].reset_index(drop=True)
        random_pairs = pd.read_csv(os.path.join(self.data_path, 'random_pairs.txt'), sep='\t', header=0)


        # seed random state from time
        random_state2 = np.random.RandomState(int(time.time()))
        ## split training set according to the given batch size
        N = train_pairs.shape[0]//self.batch_size
        # Sets up 10-fold cross validation set
        cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)
        train_batch = []
        try:
            os.mkdir(os.path.join(self.processed_dir, 'train_loaders'))
        except FileExistsError:
            pass
        for _, index in cv2.split(np.array(list(train_pairs.index)), np.array(train_pairs['y'])):
            train_batch += [train_pairs.loc[list(index),:].reset_index(drop=True)]

        self.logger.info("generating batches for train set")
        for i in trange(len(train_batch)):

            batch_data = train_batch[i]
            data_set = set()
            data_set.update(set(batch_data.source))
            data_set.update(set(batch_data.target))
            data_idx = torch.tensor(list(map(entity2id.get, data_set)), dtype=torch.int32) - 1
            for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
                adjs = [(adj.edge_index,adj.size) for adj in adjs]
                loader = (n_id, adjs)
            filename = 'train_loader' + '_' + str(i) + '.pkl'
            with open(os.path.join(self.processed_dir, 'train_loaders', filename), 'wb') as output:
                pickle.dump(loader, output)
        
        
        ## split validation set according to the given batch size
        N = val_pairs.shape[0]//self.batch_size
        # Sets up 10-fold cross validation set
        cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)
        val_batch = []
        try:
            os.mkdir(os.path.join(self.processed_dir, 'val_loaders'))
        except FileExistsError:
            pass
        for _, index in cv2.split(np.array(list(val_pairs.index)), np.array(val_pairs['y'])):
            val_batch += [val_pairs.loc[list(index),:].reset_index(drop=True)]

        self.logger.info("generating batches for validation set")
        for i in trange(len(val_batch)):

            batch_data = val_batch[i]
            data_set = set()
            data_set.update(set(batch_data.source))
            data_set.update(set(batch_data.target))
            data_idx= torch.tensor(list(map(entity2id.get, data_set)), dtype=torch.int32) - 1
            for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
                adjs = [(adj.edge_index,adj.size) for adj in adjs]
                loader = (n_id, adjs)
            filename = 'val_loader' + '_' + str(i) + '.pkl'
            with open(os.path.join(self.processed_dir, 'val_loaders', filename), 'wb') as output:
                pickle.dump(loader, output)            
            
        
        ## split test set according to the given batch size
        N = test_pairs.shape[0]//self.batch_size
        # Sets up 10-fold cross validation set
        cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)    
        test_batch = []
        try:
            os.mkdir(os.path.join(self.processed_dir, 'test_loaders'))
        except FileExistsError:
            pass
        for _, index in cv2.split(np.array(list(test_pairs.index)), np.array(test_pairs['y'])):
            test_batch += [test_pairs.loc[list(index),:].reset_index(drop=True)]

        self.logger.info("generating batches for test set")
        for i in trange(len(test_batch)):

            batch_data = test_batch[i]
            data_set = set()
            data_set.update(set(batch_data.source))
            data_set.update(set(batch_data.target))
            data_idx= torch.tensor(list(map(entity2id.get, data_set)), dtype=torch.int32) - 1
            for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
                adjs = [(adj.edge_index,adj.size) for adj in adjs]
                loader = (n_id, adjs)
            filename = 'test_loader' + '_' + str(i) + '.pkl'
            with open(os.path.join(self.processed_dir, 'test_loaders', filename), 'wb') as output:
                pickle.dump(loader, output)

                
        ## split random pair set according to the given batch size
        N = random_pairs.shape[0]//(self.batch_size * 3)
        # Sets up 10-fold cross validation set
        cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)    
        random_batch = []
        try:
            os.mkdir(os.path.join(self.processed_dir, 'random_loaders'))
        except FileExistsError:
            pass
        for _, index in cv2.split(np.array(list(random_pairs.index)), np.array(random_pairs['y'])):
            random_batch += [random_pairs.loc[list(index),:].reset_index(drop=True)]

        self.logger.info("generating batches for random pair set")
        for i in trange(len(random_batch)):

            batch_data = random_batch[i]
            data_set = set()
            data_set.update(set(batch_data.source))
            data_set.update(set(batch_data.target))
            data_idx=torch.tensor(list(map(entity2id.get, data_set)), dtype=torch.int32) - 1
            for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
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

    def get_test_loader(self):
        test_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'test_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return DataLoader(eval_utilities.DataWrapper(test_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)

    def get_random_loader(self):
        random_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'random_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return DataLoader(eval_utilities.DataWrapper(random_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)

