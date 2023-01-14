import pandas as pd
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
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

class DataWrapper(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        n_id, adjs = pickle.load(open(self.paths[idx],'rb'))
        return (n_id, adjs)

class ProcessedDataset(InMemoryDataset):
    def __init__(self, root, args, batch_size=128, layers=2, dim=100, num_samples=None, logger=None):

        self.args = args
        self.batch_size = batch_size
        self.dim = dim
        self.logger = logger
        self.worker = 8
        self.layer_size = []
        if num_samples is None:
            for _ in range(layers):
                self.layer_size += [-1]
        else:
            self.layer_size = num_samples

        super(ProcessedDataset, self).__init__(root, transform=None, pre_transform=None)

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['processed_kg.dataset', 'map_files.pkl', 'train_val_test_random.pkl']

    def download(self):
        pass
    
    @staticmethod
    def _encode_onehot(labels):
        ulabels = set(labels)
        ulabels_dict = {c: list(np.identity(len(ulabels))[i, :]) for i, c in enumerate(ulabels)}
        return (np.array(list(map(ulabels_dict.get, labels)), dtype=np.int32), ulabels_dict)

    # @staticmethod
    # def _generate_init_emb(entity2id, id2entity, dim=100, known_int_emb_dict=None):

    #     if known_int_emb_dict:
    #         init_embs = [known_int_emb_dict[id2entity[entity_id]] if entity_id != 0 else np.zeros(len(known_int_emb_dict[id2entity[1]])) for entity_id in id2entity]
    #         init_embs = torch.tensor(np.vstack(init_embs), dtype=torch.float32)
    #     else:
    #         init_embs = torch.nn.Embedding(len(entity2id), dim)
    #         init_embs.requires_grad = False
        
    #     return init_embs

    @staticmethod
    def _generate_init_emb(idx_map, node_info, dim=100, known_int_emb_dict=None):
        init_embs = dict()
        ulabels = list(set(node_info.category))
        if known_int_emb_dict is not None:
            known_int_emb_df = pd.DataFrame([(curie_id, array) for curie_id, array in known_int_emb_dict.items()]).rename(columns={0:'id',1:'array'})
            known_int_emb_df = known_int_emb_df.merge(node_info,on='id').reset_index(drop=True)
            category_has_known_init_emb = set(known_int_emb_df['category'])
            for category in category_has_known_init_emb:
                try:
                    assert known_int_emb_df.loc[known_int_emb_df.category.isin([category]),:].shape[0] == node_info.loc[node_info.category.isin([category]),:].shape[0]
                except AssertionError:
                    print(f"Not all curies with cateogry {category} have known intial embedding")
                curie_ids = known_int_emb_df.loc[known_int_emb_df.category.isin([category]),'id']
                curie_ids = torch.tensor(list(map(idx_map.get, curie_ids)))
                init_emb = torch.tensor(np.vstack(list(known_int_emb_df.loc[known_int_emb_df.category.isin([category]),'array'])).astype(float), dtype=torch.float32)
                init_embs[category] = (init_emb, curie_ids)
                ulabels.remove(category)
            
            print(f"Number of categories that are not uninitialized with known embeddings: {len(ulabels)}")

        for label in ulabels:
            curie_ids = node_info.loc[node_info.category.isin([label]),'id']
            curie_ids = torch.tensor(list(map(idx_map.get, curie_ids)))
            init_emb = torch.normal(0, 1, size=(len(curie_ids), dim), dtype=torch.float32)
            init_embs[label] = (init_emb, curie_ids)

        return init_embs

    def process(self):

        raw_edges = pd.read_csv(os.path.join(self.args.data_dir,'graph_edges.txt'), sep='\t', header=0)[['source','target']]
        node_info = pd.read_csv(os.path.join(self.args.data_dir,'all_graph_nodes_info.txt'), sep='\t', header=0)

        if self.args.use_known_embedding:
            if not os.path.exists(self.args.known_embedding):
                self.logger.Warning(f"Not found {self.args.known_embedding}. Set 'known_int_emb_dict' to None")
                known_int_emb_dict = None
            else:
                with open(self.args.known_embedding, 'rb') as file_in:
                    known_int_emb_dict = pickle.load(file_in)
        else:
            known_int_emb_dict = None

        ## read node mapping
        entity2id, _ = utils.load_index(os.path.join(self.args.data_dir, 'entity2freq.txt'))
        idx_map = {name:idx-1 for name, idx in entity2id.items() if idx !=0}

        ## filter out idx not in idx_map
        node_info = node_info.loc[node_info['id'].isin(idx_map.keys()),:].reset_index(drop=True)

        ## convert edge to idx link
        edges = np.array(list(map(idx_map.get, np.array(raw_edges).flatten())), dtype=np.int32).reshape(np.array(raw_edges).shape)

        ## generate initial embedding vectors
        init_embs = self._generate_init_emb(idx_map, node_info, dim=self.dim, known_int_emb_dict=known_int_emb_dict)
        ## generate edge index matrix
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        data = Data(feat=init_embs, edge_index=edge_index)
        id_to_type = {idx_map[node_info['id'][index]]:node_info['category'][index] for index in range(node_info.shape[0])}
        typeid = {key:index for index, key in enumerate(init_embs)}
        map_files = [idx_map, id_to_type, typeid]

        ## split dataset to training, validation and test according 
        # seed random state from time
        ### generate train, validation and test pairs
        self.logger.info(f"read training, validation and test pairs")
        data_list = []
        for dataset in ['train', 'val', 'test', 'random']:
            data_list.append(pd.read_csv(os.path.join(self.args.data_dir, 'pretrain_reward_shaping_model_train_val_test_random_data_2class', f'{dataset}_pairs.txt'), sep='\t', header=0))

        ## prepare the feature vectors
        train_pairs = data_list[0]
        val_pairs = data_list[1]
        test_pairs = data_list[2]
        random_pairs = data_list[3]

        if not os.path.exists(os.path.join(self.processed_dir, 'train_loaders')):
            ## split training set according to the given batch size
            N = train_pairs.shape[0]//self.batch_size
            # seed random state from time
            random_state2 = np.random.RandomState(int(time.time()))
            cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)
            train_batch = list()
            if not os.path.exists(os.path.join(self.processed_dir, 'train_loaders')):
                os.makedirs(os.path.join(self.processed_dir, 'train_loaders'))

            for _, index in cv2.split(np.array(list(train_pairs.index)), np.array(train_pairs['y'])):
                train_batch += [train_pairs.loc[list(index),:].reset_index(drop=True)]

            self.logger.info(f"generating batches for training set")
            for i in trange(len(train_batch)):

                batch_data = train_batch[i]
                data_set = set()
                data_set.update(set(batch_data.source))
                data_set.update(set(batch_data.target))
                data_idx=torch.tensor(list(map(idx_map.get, data_set)), dtype=torch.int32)
                for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
                    adjs = [(adj.edge_index,adj.size) for adj in adjs]
                    loader = (n_id, adjs)
                filename = 'train_loader' + '_' + str(i) + '.pkl'
                with open(os.path.join(self.processed_dir, 'train_loaders', filename), 'wb') as output:
                    pickle.dump(loader, output)

        if not os.path.exists(os.path.join(self.processed_dir, 'val_loaders')):
            ## split validation set according to the given batch size
            N = val_pairs.shape[0]//self.batch_size
            # seed random state from time
            random_state2 = np.random.RandomState(int(time.time()))
            cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)
            val_batch = list()
            if not os.path.exists(os.path.join(self.processed_dir, 'val_loaders')):
                os.makedirs(os.path.join(self.processed_dir, 'val_loaders'))

            for _, index in cv2.split(np.array(list(val_pairs.index)), np.array(val_pairs['y'])):
                val_batch += [val_pairs.loc[list(index),:].reset_index(drop=True)]

            self.logger.info(f"generating batches for validation set")
            for i in trange(len(val_batch)):

                batch_data = val_batch[i]
                data_set = set()
                data_set.update(set(batch_data.source))
                data_set.update(set(batch_data.target))
                data_idx=torch.tensor(list(map(idx_map.get, data_set)), dtype=torch.int32)
                for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
                    adjs = [(adj.edge_index,adj.size) for adj in adjs]
                    loader = (n_id, adjs)
                filename = 'val_loader' + '_' + str(i) + '.pkl'
                with open(os.path.join(self.processed_dir, 'val_loaders', filename), 'wb') as output:
                    pickle.dump(loader, output)


        if not os.path.exists(os.path.join(self.processed_dir, 'test_loaders')):
            ## split test set according to the given batch size
            N = test_pairs.shape[0]//self.batch_size
            # seed random state from time
            random_state2 = np.random.RandomState(int(time.time()))
            cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)
            test_batch = list()
            if not os.path.exists(os.path.join(self.processed_dir, 'test_loaders')):
                os.makedirs(os.path.join(self.processed_dir, 'test_loaders'))

            for _, index in cv2.split(np.array(list(test_pairs.index)), np.array(test_pairs['y'])):
                test_batch += [test_pairs.loc[list(index),:].reset_index(drop=True)]

            self.logger.info("generating batches for test set")
            for i in trange(len(test_batch)):

                batch_data = test_batch[i]
                data_set = set()
                data_set.update(set(batch_data.source))
                data_set.update(set(batch_data.target))
                data_idx=torch.tensor(list(map(idx_map.get, data_set)), dtype=torch.int32)
                for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
                    adjs = [(adj.edge_index,adj.size) for adj in adjs]
                    loader = (n_id, adjs)
                filename = 'test_loader' + '_' + str(i) + '.pkl'
                with open(os.path.join(self.processed_dir, 'test_loaders', filename), 'wb') as output:
                    pickle.dump(loader, output)

        if not os.path.exists(os.path.join(self.processed_dir, 'random_loaders')):
            ## split random pair set according to the given batch size
            N = random_pairs.shape[0]//self.batch_size
            # seed random state from time
            random_state2 = np.random.RandomState(int(time.time()))
            # Sets up 10-fold cross validation set
            cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)    
            random_batch = list()
            if not os.path.exists(os.path.join(self.processed_dir, 'random_loaders')):
                os.makedirs(os.path.join(self.processed_dir, 'random_loaders'))

            for _, index in cv2.split(np.array(list(random_pairs.index)), np.array(random_pairs['y'])):
                random_batch += [random_pairs.loc[list(index),:].reset_index(drop=True)]

            print(f"generating batches for random pair set", flush=True)
            for i in trange(len(random_batch)):

                batch_data = random_batch[i]
                data_set = set()
                data_set.update(set(batch_data.source))
                data_set.update(set(batch_data.target))
                data_idx=torch.tensor(list(map(idx_map.get, data_set)), dtype=torch.int32)
                for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
                    adjs = [(adj.edge_index,adj.size) for adj in adjs]
                    loader = (n_id, adjs)
                filename = 'random_loader' + '_' + str(i) + '.pkl'
                with open(os.path.join(self.processed_dir, 'random_loaders', filename), 'wb') as output:
                    pickle.dump(loader, output)

        if not os.path.exists(os.path.join(self.processed_dir, 'train_val_test_random.pkl')):
            train_val_test_random = [train_batch, val_batch, test_batch, random_batch]

            with open(os.path.join(self.processed_dir, 'train_val_test_random.pkl'), 'wb') as output:
                pickle.dump(train_val_test_random, output)

        if not os.path.exists(os.path.join(self.processed_dir, 'map_files.pkl')):
            with open(os.path.join(self.processed_dir, 'map_files.pkl'), 'wb') as output:
                pickle.dump(map_files, output)

        if not os.path.exists(os.path.join(self.processed_dir, 'processed_kg.dataset')):
            torch.save(data, os.path.join(self.processed_dir, 'processed_kg.dataset'))

    def get_dataset(self):
        data = torch.load(os.path.join(self.processed_dir, 'processed_kg.dataset'))
        return data

    def get_train_val_test_random(self):
        train_val_test_random = pickle.load(open(os.path.join(self.processed_dir, 'train_val_test_random.pkl'), 'rb'))
        return train_val_test_random

    def get_mapfiles(self):
        mapfiles = pickle.load(open(os.path.join(self.processed_dir, 'map_files.pkl'), 'rb'))
        return mapfiles

    def get_train_loader(self):
        train_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'train_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return DataLoader(DataWrapper(train_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)

    def get_val_loader(self):
        val_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'val_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return DataLoader(DataWrapper(val_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)

    def get_test_loader(self):
        test_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'test_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return DataLoader(DataWrapper(test_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)

    def get_random_loader(self):
        random_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'random_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return DataLoader(DataWrapper(random_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)
