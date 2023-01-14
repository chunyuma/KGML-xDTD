import os
import csv
import pickle
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    "--graph_data_path",
    default='./data/raw',
    type=str,
    help="Path to the folder that contain graph node/edge information",
)
parser.add_argument(
    "--pair_data_path",
    default='./data/pretrain_reward_shaping_model_train_val_test_random_data_2class/',
    type=str,
    help="Path to the folder that contain information about treat/random/not_treat drug-disease pairs",
)
parser.add_argument(
    "--output_path",
    default='./results/processed',
    type=str,
    help="Path to the processed data (in OpenKE format)",
)
args = parser.parse_args()

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

def load_index(input_path):
    name_to_id, id_to_name = {}, {}
    with open(input_path, 'r') as f:
        for index, line in enumerate(f.readlines()):
            name, _ = line.strip().split()
            name_to_id[name] = index
            id_to_name[index] = name
    return name_to_id, id_to_name

entity2id, id2entity = load_index(args.graph_data_path + 'entity2freq.txt')
relation2id, id2relation = load_index(args.graph_data_path + 'relation2freq.txt')
tpID = relation2id['biolink:has_effect']
tnID = relation2id['biolink:has_no_effect']

with open(args.output_path + 'relation2id.txt', 'w') as f:
    f.write(str(len(relation2id)))
    f.write('\n')
    for key in relation2id:
        f.write(key + "\t" + str(relation2id[key]) + "\n")
with open(args.output_path + 'entity2id.txt', 'w') as f:
    f.write(str(len(entity2id)))
    f.write('\n')
    for key in entity2id:
        f.write(key + "\t" + str(entity2id[key]) + "\n")



with open(args.graph_data_path + 'graph_edges.txt', 'r') as f:
    train_lines = list(csv.reader(f, delimiter="\t"))[1:]
    train_lines = [[entity2id[l[0]], entity2id[l[1]], relation2id[l[2]]] for l in train_lines]


def load_pair(input_path):
    lines = []
    with open(input_path, 'r') as f:
        pairs = list(csv.reader(f, delimiter="\t"))[1:]
        for p in pairs:
            source = entity2id[p[0]]
            target = entity2id[p[1]]
            relation = tpID if int(p[2]) == 1 else tnID
            lines.append([source, target, relation])
        
    return lines

train_lines_tptn = load_pair(args.pair_data_path + 'train_pairs.txt')
train_lines.extend(train_lines_tptn)
val_lines = load_pair(args.pair_data_path + 'val_pairs.txt')
test_lines = load_pair(args.pair_data_path + 'test_pairs.txt')


def write_file(lines, output_file):
    with open(output_file, 'w') as f:
        f.write(str(len(lines)) + "\n")
        f_w = csv.writer(f, delimiter="\t")
        for line in lines:
            f_w.writerow(line)

write_file(train_lines, args.output_path + 'train2id.txt')
write_file(val_lines, args.output_path + 'valid2id.txt')
write_file(test_lines, args.output_path + 'test2id.txt')


# get the name of all diseases and drugs
def load_index(input_path):
    name_to_id, id_to_name = {}, {}
    with open(input_path) as f:
        for index, line in enumerate(f.readlines()):
            name, _ = line.strip().split()
            name_to_id[name] = index
            id_to_name[index] = name
    return name_to_id, id_to_name

entity2id, id2entity = load_index(os.path.join(args.graph_data_path, 'entity2freq.txt'))

type2id, id2type = load_index(os.path.join(args.graph_data_path, 'type2freq.txt'))
with open(os.path.join(args.graph_data_path, 'entity2typeid.pkl'), 'rb') as infile:
    entity2typeid = pickle.load(infile)
drug_type = ['biolink:Drug', 'biolink:SmallMolecule']
drug_type_ids = [type2id[x] for x in drug_type]
all_drug_ids = [id2entity[index] for index, typeid in enumerate(entity2typeid) if typeid in drug_type_ids]
disease_type = ['biolink:Disease', 'biolink:PhenotypicFeature', 'biolink:BehavioralFeature', 'biolink:DiseaseOrPhenotypicFeature']
disease_type_ids = [type2id[x] for x in disease_type]
all_disease_ids = [id2entity[index] for index, typeid in enumerate(entity2typeid) if typeid in disease_type_ids]

with open(args.output_path + "all_drug_ids.json", "w") as f:
    json.dump(all_drug_ids, f)

with open(args.output_path + "all_disease_ids.json", "w") as f:
    json.dump(all_disease_ids, f)