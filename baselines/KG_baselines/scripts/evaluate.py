import os
import json
import time
import pickle
import pandas as pd
import numpy as np
from torch import tensor
import torch
import openke
from openke.module.model import TransE, RotatE, DistMult, ComplEx, TransR, SimplE, Analogy
from sklearn.metrics import f1_score, average_precision_score, classification_report
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    "--data",
    default='data/processed/',
    type=str,
    help="The dir where train/test/val files locates",
)
parser.add_argument(
    "--pair_data_path",
    default='./data/pretrain_reward_shaping_model_train_val_test_random_data_2class/',
    type=str,
    help="Path to the folder that contain information about treat/random/not_treat drug-disease pairs",
)
parser.add_argument(
    "--checkpoint",
    default='./results/checkpoints/',
    type=str,
    help="Output Checkpoint Folder",
)
args = parser.parse_args()


model2dim = {
    'transe': 100,
    'distmult': 100,
    'simple': 100,
    'rotate': 30,
    'complex': 50,
    'transr': 50,
    'analogy': 20,
}


all_models = os.listdir(args.checkpoint)
all_models = [m for m in all_models if m.endswith('.ckpt')]

print("Loading entity2id and relation2id")
with open(os.path.join(args.args, "entity2id.txt"), "r") as f:
    entity2id_raw = f.readlines()
with open(os.path.join(args.args, "relation2id.txt"), "r") as f:
    relation2id_raw = f.readlines()

entity2id = {}
relation2id = {}
for line in entity2id_raw[1:]:
    line = line.strip("\n").split("\t")
    entity2id[line[0]] = line[1]
for line in relation2id_raw[1:]:
    line = line.strip("\n").split("\t")
    relation2id[line[0]] = line[1]

with open(os.path.join(args.args, "all_disease_ids.json"), "r") as f:
    all_disease_ids = json.load(f)
    all_disease_ids = [float(entity2id[item]) for item in all_disease_ids]
with open(os.path.join(args.args, "all_drug_ids.json"), "r") as f:
    all_drug_ids = json.load(f)
    all_drug_ids = [float(entity2id[item]) for item in all_drug_ids]
num_diseases = len(all_disease_ids)
num_drugs = len(all_drug_ids)


print("Loading data from {}".format(args.args))
with open(os.path.join(args.args, 'valid2id.txt'), 'r') as f:
    valid2id = f.readlines()[1:]
    valid2id = [l.strip("\n").split("\t") for l in valid2id] 
    sources = [int(l[0]) for l in valid2id]
    targets = [int(l[1]) for l in valid2id]
    relations = [int(l[2]) for l in valid2id]

with open(os.path.join(args.args, 'test2id.txt'), 'r') as f:
    test2id = f.readlines()[1:]
    test2id = [l.strip("\n").split("\t") for l in test2id] 
    sources += [int(l[0]) for l in test2id]
    targets += [int(l[1]) for l in test2id]
    relations += [int(l[2]) for l in test2id]


random_pairs = pd.read_csv(os.path.join(args.pair_data_path, 'random_pairs.txt'), delimiter='\t' )
train_pairs = pd.read_csv(os.path.join(args.pair_data_path, 'train_pairs.txt'), delimiter='\t' ) 
valid_pairs = pd.read_csv(os.path.join(args.pair_data_path, 'val_pairs.txt'), delimiter='\t' ) 
test_pairs = pd.read_csv(os.path.join(args.pair_data_path, 'test_pairs.txt'), delimiter='\t' )
train_tp_pairs = train_pairs.loc[train_pairs['y'] == 1].reset_index(drop=True)
test_tp_pairs = test_pairs.loc[test_pairs['y'] == 1].reset_index(drop=True)
# test_tp_pairs = pd.concat([valid_pairs.loc[valid_pairs['y'] == 1].reset_index(drop=True),  test_pairs.loc[test_pairs['y'] == 1].reset_index(drop=True)], ignore_index=True) 
all_tp_pairs = pd.concat([train_tp_pairs, test_tp_pairs], ignore_index=True) 

for model in all_models:
    MODEL_TYPE = model.split("_")[0]
    dim = model2dim[MODEL_TYPE]
    MODEL_DIR = os.path.join(args.checkpoint, model)

    print("Loading model from {}".format(MODEL_DIR))
    if MODEL_TYPE == "transe":
        kg_model = TransE(
            ent_tot = len(entity2id),
            rel_tot = len(relation2id),
            dim = dim, 
            p_norm = 1, 
            norm_flag = True)
    elif MODEL_TYPE == "rotate":
        kg_model = RotatE(
        ent_tot = len(entity2id),
        rel_tot = len(relation2id),
        dim = dim,
        margin = 6.0,
        epsilon = 2.0,
    )
    elif MODEL_TYPE == "complex":
        kg_model = ComplEx(
        ent_tot = len(entity2id),
        rel_tot = len(relation2id),
        dim = dim)
    elif MODEL_TYPE == 'distmult':
        kg_model = DistMult(
            ent_tot = len(entity2id),
            rel_tot = len(relation2id),
            dim = dim
        )
    elif MODEL_TYPE == 'transr':
        kg_model = TransR(
            ent_tot = len(entity2id),
            rel_tot = len(relation2id),
            dim_e = dim,
            dim_r = dim,
            p_norm = 1, 
            norm_flag = True,
            rand_init = True)
    elif MODEL_TYPE == 'simple':
        kg_model = SimplE(
            ent_tot = len(entity2id),
            rel_tot = len(relation2id),
            dim = dim
        )
    elif MODEL_TYPE == 'analogy':
        kg_model = Analogy(
            ent_tot = len(entity2id),
            rel_tot = len(relation2id),
            dim = dim
        )

    kg_model.load_checkpoint(MODEL_DIR)
    kg_model.cuda()




    print("Results of {}:".format(MODEL_TYPE))



    def pred_acc(sources, targets):
        num_sample = 2 * len(sources) 
        data = {}
        data['batch_h'] = torch.zeros([num_sample])
        data['batch_t'] = torch.zeros([num_sample])
        data['batch_r'] = torch.zeros([num_sample]) 
        data['batch_r'][:len(sources)] = int(relation2id['biolink:has_effect'])
        data['batch_r'][len(sources):] = int(relation2id['biolink:has_no_effect'])
        data['mode'] = 'normal'
        for i, (source, target) in enumerate(zip(sources, targets)):
            s_id = source
            t_id = target
            data['batch_h'][i] = float(s_id)
            data['batch_h'][i+len(sources)] = float(s_id)
            data['batch_t'][i] = float(t_id)
            data['batch_t'][i+len(sources)] = float(t_id)
        
        data['batch_h'] = data['batch_h'].long().cuda()
        data['batch_t'] = data['batch_t'].long().cuda()
        data['batch_r'] = data['batch_r'].long().cuda()
        
        return kg_model.predict(data)

    def softmax(x):
        return np.exp(x, axis=1)/sum(np.exp(x))

    num_samples = len(sources)
    acc_res = pred_acc(sources, targets)
    scores = np.zeros([num_samples, 2])
    scores[:,0] = acc_res[:num_samples]
    scores[:,1] = acc_res[num_samples:]

    preds = np.zeros([num_samples])
    for i in range(num_samples):
        preds[i] = 1 if scores[i][0] < scores[i][1] else 0

    labels = np.array([1 if r == int(relation2id['biolink:has_effect']) else 0 for r in relations])
    accuracy = (preds == labels).sum() / num_samples
    binary_f1 = f1_score(labels, preds, average='binary')
    micro_f1 = f1_score(labels, preds, average='micro')
    macro_f1 = f1_score(labels, preds, average='macro')
    weighted_f1 = f1_score(labels, preds, average='weighted')
    # ap = average_precision_score(labels, preds)
    my_classification_report = classification_report(labels, preds)
    print("Accuracy: {}".format(accuracy))
    print("Binary F1 score: {}".format(binary_f1))
    print("Micro F1 score: {}".format(micro_f1))
    print("Macro F1 score: {}".format(macro_f1))
    print("Weighted F1 score: {}".format(weighted_f1))
    print(my_classification_report)



    #### for mrr and hit@k with 500 + 500
    def pred_mrr(sources, targets):
        data = {}
        data['batch_h'] = torch.Tensor(sources)
        data['batch_t'] = torch.Tensor(targets)
        data['batch_r'] = torch.ones([len(sources)]) * int(relation2id["biolink:has_effect"])
        data['mode'] = 'normal'

        data['batch_h'] = data['batch_h'].long().cuda()
        data['batch_t'] = data['batch_t'].long().cuda()
        data['batch_r'] = data['batch_r'].long().cuda()
        
        return kg_model.predict(data)


    # Q_n = len(test_tp_pairs)
    # count_1 = 0
    # count_2 = 0
    # count_3 = 0
    # count_4 = 0
    # count_5 = 0
    # count_6 = 0
    # count_7 = 0
    # count_8 = 0
    # count_9 = 0
    # count_10 = 0
    # count_20 = 0 
    # count_50 = 0
    # count_mrr = 0


    # for index in range(Q_n):
    #     query_drug = test_tp_pairs['source'][index]
    #     query_disease = test_tp_pairs['target'][index]
    #     random_sources = []
    #     random_targets = []
    #     random_sources.append(test_tp_pairs['source'][index])
    #     random_targets.append(test_tp_pairs['target'][index])
    #     random_drugs = random_pairs.loc[random_pairs['source'].isin([query_drug])]
    #     random_diseases = random_pairs.loc[random_pairs['target'].isin([query_disease])]
    #     for i, row in random_drugs.iterrows():
    #         random_sources.append(row['source'])
    #         random_targets.append(row['target']) 
    #     for i, row in random_diseases.iterrows():
    #         random_sources.append(row['source'])
    #         random_targets.append(row['target']) 

    #     all_res = pred_mrr(random_sources, random_targets)
    #     this_query_score = float(all_res[0])
    #     all_random_probs_for_this_query = list(all_res[1:])
    #     all_in_list = [this_query_score] + all_random_probs_for_this_query
    #     rank = list(tensor(all_in_list).sort(descending=False).indices.numpy()).index(0)+1

    #     if rank <= 1:
    #         count_1 += 1 
    #     if rank <= 2:
    #         count_2 += 1        
    #     if rank <= 3:
    #         count_3 += 1 
    #     if rank <= 4:
    #         count_4 += 1
    #     if rank <= 5:
    #         count_5 += 1    
    #     if rank <= 6:
    #         count_6 += 1 
    #     if rank <= 7:
    #         count_7 += 1
    #     if rank <= 8:
    #         count_8 += 1         
    #     if rank <= 9:
    #         count_9 += 1 
    #     if rank <= 10:
    #         count_10 += 1 
    #     if rank <= 20:
    #         count_20 += 1 
    #     if rank <= 50:
    #         count_50 += 1 

    #     count_mrr += 1/rank

    # print("MRR: {}".format(count_mrr/Q_n))
    # print("Hit@1: {}".format(count_1/Q_n))
    # print("Hit@2: {}".format(count_2/Q_n))
    # print("Hit@3: {}".format(count_3/Q_n))
    # print("Hit@4: {}".format(count_4/Q_n))
    # print("Hit@5: {}".format(count_5/Q_n))
    # print("Hit@6: {}".format(count_6/Q_n))
    # print("Hit@7: {}".format(count_7/Q_n))
    # print("Hit@8: {}".format(count_8/Q_n))
    # print("Hit@9: {}".format(count_9/Q_n))
    # print("Hit@10: {}".format(count_10/Q_n))
    # print("Hit@20: {}".format(count_20/Q_n))
    # print("Hit@50: {}".format(count_50/Q_n))


    #### for mrr and hit@k with all other drugs and diseases
    Q_n = len(test_tp_pairs)
    count = {}
    for key in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "20", "30", "50"]:
        count[key] = [0, 0, 0]
    count_mrr = [0, 0, 0]

    start_time = time.time()

    for index in range(Q_n):
        query_drug = float(entity2id[test_tp_pairs['source'][index]])
        query_disease = float(entity2id[test_tp_pairs['target'][index]])
        random_sources = []
        random_targets = []
        random_sources.append(query_drug)
        random_targets.append(query_disease)
        
        cur_all_drug_ids = all_drug_ids.copy()
        cur_all_disease_ids = all_disease_ids.copy()

        cur_all_drug_ids.remove(query_drug)
        cur_all_disease_ids.remove(query_disease)

        random_drugs = cur_all_drug_ids + [query_drug]*(num_diseases-1)
        random_diseases = [query_disease]*(num_drugs-1) + cur_all_disease_ids
        random_sources.extend(random_drugs)
        random_targets.extend(random_diseases)

        res_all = pred_mrr(random_sources, random_targets)
        res_all = list(res_all)
        res_other = res_all[1:]
        res_drug = res_all[1:num_drugs]
        res_disease = res_all[num_drugs:]

        this_query_score = float(res_all[0])
        list_all = [this_query_score] + res_other
        list_drug = [this_query_score] + res_drug
        list_disease = [this_query_score] + res_disease
        rank_all = list(tensor(list_all).sort(descending=False).indices.numpy()).index(0)+1
        rank_drug = list(tensor(list_drug).sort(descending=False).indices.numpy()).index(0)+1
        rank_disease = list(tensor(list_disease).sort(descending=False).indices.numpy()).index(0)+1

        for key in count:
            if rank_all <= int(key):
                count[key][0] += 1
            if rank_drug <= int(key):
                count[key][1] += 1
            if rank_disease <= int(key):
                count[key][2] += 1

        count_mrr[0] += 1/rank_all
        count_mrr[1] += 1/rank_drug
        count_mrr[2] += 1/rank_disease

    print("Time: ", time.time() - start_time)

    for i in range(3):
        print("MRR: {}".format(count_mrr[i]/Q_n))
        print("Hit@1: {}".format(count["1"][i]/Q_n))
        print("Hit@2: {}".format(count["2"][i]/Q_n))
        print("Hit@3: {}".format(count["3"][i]/Q_n))
        print("Hit@4: {}".format(count["4"][i]/Q_n))
        print("Hit@5: {}".format(count["5"][i]/Q_n))
        print("Hit@6: {}".format(count["6"][i]/Q_n))
        print("Hit@7: {}".format(count["7"][i]/Q_n))
        print("Hit@8: {}".format(count["8"][i]/Q_n))
        print("Hit@9: {}".format(count["9"][i]/Q_n))
        print("Hit@10: {}".format(count["10"][i]/Q_n))
        print("Hit@20: {}".format(count["20"][i]/Q_n))
        print("Hit@30: {}".format(count["30"][i]/Q_n))
        print("Hit@50: {}".format(count["50"][i]/Q_n))
        print("\n\n")
