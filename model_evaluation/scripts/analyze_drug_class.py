import sys, os
import eval_utilities
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from QueryMyChem import QueryMyChem
from KGML_xDTD import KGML_xDTD
import pickle

def check_train_and_drug_class(df, disease_curie, known_drug_class_set):
    drug_class_temp = []
    is_in_tarin_temp = []
    known_drug_class_temp = []
    for row in tqdm(df.to_numpy(), desc='Get drug class for top 300 drugs'):
        drug_class = query_mychem.get_drug_or_chemical_class(row[0])
        drug_class_set = set()
        if type(drug_class) is str:
            drug_class_set.add(drug_class)
        else:
            drug_class_set.update(set([x[0] for x in drug_class]))
        drug_class_temp.append(drug_class_set)
        is_in_tarin_temp.append((row[0], disease_curie) in train_drug_disease_pairs)
        known_drug_class_temp.append(len(drug_class_set.intersection(known_drug_class_set)) != 0)
    df['drug_class'] = drug_class_temp
    df['is_in_tarin'] = is_in_tarin_temp
    df['known_drug_class'] = known_drug_class_temp
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="The path to data folder", default="../data")
    parser.add_argument("--model_path", type=str, help="The path to model folder", default="../models")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU to train model", default=False)
    parser.add_argument("--gpu", type=int, help='gpu device (default: 0) if use_multiple_gpu is False', default=1)
    args = parser.parse_args()

    logger = eval_utilities.get_logger()
    synonymizer = eval_utilities.nodesynonymizer
    query_mychem = QueryMyChem()
    
    ## Get the true positive pairs from training, validation, and test data
    logger.info('Get the true positive pairs from training, validation, and test data')
    data_list = []
    for dataset in ['train', 'val', 'test']:
        data = pd.read_csv(os.path.join(args.data_path, f'{dataset}_pairs.txt'), sep='\t', header=0)
        data = data.loc[data['y'] == 1, :].reset_index(drop=True)
        data['data_type'] = dataset
        data_list.append(data)
        
    ## Combine data into one single dataframe
    data = pd.concat(data_list, axis=0).reset_index(drop=True)
    data.drop(columns=['y'], inplace=True)
    data.columns = ['drug', 'disease', 'data_type']
    
    ## Get drug name and disease name
    drug_normalized_result = synonymizer.get_canonical_curies(data['drug'].tolist())
    drug_id_to_name = {key:drug_normalized_result[key]['preferred_name'] for key in drug_normalized_result if drug_normalized_result[key]}
    disease_normalized_result = synonymizer.get_canonical_curies(data['disease'].tolist())
    disease_id_to_name = {key:disease_normalized_result[key]['preferred_name'] for key in disease_normalized_result if disease_normalized_result[key]}
    
    ## Get drug class
    drug_list = list(set(data['drug'].tolist()))
    drug_class = dict()
    for drug_curie in tqdm(drug_list, desc='Get drug class'):
        drug_class[drug_curie] = query_mychem.get_drug_or_chemical_class(drug_curie)
    
    ## apply name to dataframe
    data = data.apply(lambda row: [row[0], drug_id_to_name.get(row[0], None), drug_class.get(row[0]), row[1], disease_id_to_name.get(row[1], None), row[2]], axis=1, result_type='expand')
    data.columns = ['drug', 'drug_name', 'drug_class', 'disease', 'disease_name', 'data_type']
    
    ## save data
    if not os.path.exists(os.path.join(args.data_path, "drug_class_experiment")):
        os.makedirs(os.path.join(args.data_path, "drug_class_experiment"))
    
    logger.info(f'Save data to {os.path.join(args.data_path, "drug_class_experiment", "all_true_positive_pairs.txt")}')
    data.to_csv(os.path.join(args.data_path, "drug_class_experiment", "all_true_positive_pairs.txt"), sep='\t', index=False)

    ## Save known drug class for diseases in test set
    disease_list = list(set(data.loc[data['data_type'] == 'test', 'disease'].tolist()))
    disease_dict = {}
    for disease_curie in tqdm(disease_list, desc='Known drug class from training set'):
        known_drug_class = data.loc[(data['disease'] == disease_curie) & (data['data_type'] == 'train'), 'drug_class'].tolist()
        disease_dict[disease_curie] = {}
        disease_dict[disease_curie]['known_drug_class'] = set()
        for x in known_drug_class:
            if type(x) is str:
                disease_dict[disease_curie]['known_drug_class'].add(x)
            else:
                disease_dict[disease_curie]['known_drug_class'].update(set([y[0] for y in x]))

    ## setup a KGML-xDTD object (this step needs to take around 5 minutes because its need to load the required files (e.g. KG) and trained modules)
    xdtd = KGML_xDTD(args, args.data_path, args.model_path)

    ## Predict top 500 drugs and top drugs with threshold 0.8 for each disease
    train_drug_disease_pairs = {(drug, disease):1 for drug, disease in data.loc[data['data_type'] == 'train', ['drug', 'disease']].to_numpy()}
    for disease_curie in tqdm(disease_dict, desc='Predict top drugs for each disease in test set'):
        known_drug_class_set = disease_dict[disease_curie]['known_drug_class']
        df = xdtd.predict_top_N_drugs(disease_curie=disease_curie, N=500)
        ## Get drug class for top 500 drugs
        disease_dict[disease_curie]['top_500_drugs'] = check_train_and_drug_class(df, disease_curie, known_drug_class_set)
        # ## Get drug class for top drugs with threshold 0.8
        # df = xdtd.predict_drugs_by_threshold(disease_curie=disease_curie, threshold=0.8)
        # disease_dict[disease_curie]['top_drugs_80_threshold'] = check_train_and_drug_class(df, disease_curie, known_drug_class_set)
        
    ## Save results
    logger.info(f'Save results to {os.path.join(args.data_path, "drug_class_experiment", "test_diseases_info.pkl")}')
    with open(os.path.join(args.data_path, "drug_class_experiment", "test_diseases_info.pkl"), 'wb') as f:
        pickle.dump(disease_dict, f)

