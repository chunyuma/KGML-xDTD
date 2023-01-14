import sys
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import sklearn.ensemble as ensemble
import sklearn.metrics as met
import time
import argparse
import joblib
from glob import glob
from sklearn.model_selection import GridSearchCV
import utils

def generate_X_and_y(data_df, pair_emb='concatenate'):
    
    if pair_emb == 'concatenate':
        X = np.vstack([np.hstack([entity_embeddings_dict[data_df.loc[index,'source']], entity_embeddings_dict[data_df.loc[index,'target']]]) for index in range(len(data_df))])
        y = np.array(list(data_df['y']))
        
    elif pair_emb == 'hadamard':
        X = np.vstack([entity_embeddings_dict[data_df.loc[index,'source']]*entity_embeddings_dict[data_df.loc[index,'target']] for index in range(len(data_df))])
        y = np.array(list(data_df['y']))
    
    else:
        raise TypeError("Only 'concatenate' or 'hadamard' is acceptable")
        
    return [X, y]

def evaluate(model, X, y_true, calculate_metric=True): 

    probas = model.predict_proba(X)

    if calculate_metric is True:
        
        ## calculate accuracy
        acc = utils.calculate_acc(probas,y_true)
        ## calculate macro F1 score
        macro_f1score = utils.calculate_f1score(probas,y_true,'macro')
        ## calculate micro F1 score
        micro_f1score = utils.calculate_f1score(probas,y_true,'micro')
        
        return [acc, macro_f1score, micro_f1score, y_true, probas]
    
    else:
        
        return [None, None, None, y_true, probas]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default="../log_folder")
    parser.add_argument("--log_name", type=str, help="log file name", default="step8_rf_3class.log")
    parser.add_argument("--data_dir", type=str, help="The path of data folder", default="../data")
    parser.add_argument("--pair_emb", type=str, help="The method for the pair embedding (concatenate or hadamard).", default="concatenate")
    parser.add_argument("--n_random_test_mrr_hk", type=int, help="Number of random pairs assigned to each TP drug/disease in test set for MRR and H@K", default=500)
    parser.add_argument('--seed', type=int, help='Random seed (default: 1023)', default=1023)
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default="../models")
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir,args.log_name))
    logger.info(args)
    utils.set_random_seed(args.seed)         

    ## read unsupervised GraphSage embedding vectors
    with open(os.path.join(args.data_dir, 'graphsage_output', 'unsuprvised_graphsage_entity_embeddings.pkl'), 'rb') as infile:
        entity_embeddings_dict = pickle.load(infile)

    ## read train set, validation set and test set (ignore the validation set)
    data_list = []
    for dataset in ['train', 'val', 'test', 'random']:
        data_list.append(pd.read_csv(os.path.join(args.data_dir, 'pretrain_reward_shaping_model_train_val_test_random_data_3class', f'{dataset}_pairs.txt'), sep='\t', header=0))

    ## prepare the feature vectors
    train_data = data_list[0]
    test_data = data_list[2]
    random_data = data_list[3]
    train_X, train_y = generate_X_and_y(train_data, pair_emb=args.pair_emb)
    test_X, test_y = generate_X_and_y(test_data, pair_emb=args.pair_emb)
    random_X, random_y = generate_X_and_y(random_data, pair_emb=args.pair_emb)

    ## create model folder
    folder_name = 'RF_model_3class'
    if not os.path.isdir(os.path.join(args.output_folder, folder_name)):
        os.mkdir(os.path.join(args.output_folder, folder_name))

    ## save embedding vectors
    if not os.path.exists(os.path.join(args.output_folder, folder_name, 'entity_embeddings.npy')):
        entity2id, id2entity = utils.load_index(os.path.join(args.data_dir, 'entity2freq.txt'))
        entity_embeddings = [entity_embeddings_dict[id2entity[entity_id]] if entity_id != 0 else np.zeros(len(entity_embeddings_dict[id2entity[1]])) for entity_id in id2entity] 
        with open(os.path.join(args.output_folder, folder_name, 'entity_embeddings.npy'), 'wb') as outfile:
            np.save(outfile, entity_embeddings)

    # Sets and fits Random ForestModel
    logger.info('Start training model')
    RF_model = ensemble.RandomForestClassifier(class_weight='balanced', random_state=args.seed, max_features="auto", oob_score=True, n_jobs=-1)
    param_grid = { 'max_depth' : [depth for depth in range(5,36,5)],
                   'n_estimators': [500, 1000, 1500, 2000],
                   'class_weight': ["balanced", "balanced_subsample"]
    }
    gs_rf = GridSearchCV(estimator=RF_model, param_grid=param_grid, cv= 10, scoring='f1_macro', return_train_score=True)
    gs_rf.fit(train_X, train_y)
    logger.info(f"The best hyper-parameter set of RF model based on gridseaerchCV is {gs_rf.best_estimator_}")
    best_rf = gs_rf.best_estimator_
    # save grid search cv results
    with open(os.path.join(args.output_folder, folder_name, 'grid_search_cv_results.pkl'), 'wb') as outfile:
        pickle.dump(gs_rf.cv_results_, outfile)
    fitModel = best_rf.fit(train_X, train_y)

    # saves the model
    model_name = f'RF_model.pt'
    joblib.dump(fitModel, os.path.join(args.output_folder, folder_name, model_name))

    logger.info("")
    logger.info('#### Evaluate best model ####')
    train_acc, train_macro_f1score, train_micro_f1score, train_y_true, train_y_probs = evaluate(fitModel, train_X, train_y)
    test_acc, test_macro_f1score, test_micro_f1score, test_y_true, test_y_probs = evaluate(fitModel, test_X, test_y)

    train_data['prob'] = train_y_probs[:,1]
    train_data['pred_y'] = np.argmax(train_y_probs, axis=1) 
    test_data['prob'] = test_y_probs[:,1]
    test_data['pred_y'] = np.argmax(test_y_probs, axis=1)
    _, _, _, random_y_true, random_y_probs = evaluate(fitModel, random_X, random_y, False)
    random_data['prob'] = random_y_probs[:,1]
    random_data['pred_y'] = np.argmax(random_y_probs, axis=1)

    with open(os.path.join(args.output_folder, folder_name, f'RF_results.pkl'), 'wb') as outfile:
        pickle.dump([train_data,test_data,random_data], outfile)

    all_evaluation_results = dict()
    all_evaluation_results['evaluation_hit_at_k_score'] = []
    test_mrr_score = utils.calculate_mrr(test_data, random_data, N=args.n_random_test_mrr_hk*2)
    logger.info(f'Accuracy: Train Accuracy: {train_acc:.5f}, Test dataset Accuracy: {test_acc:.5f}')
    logger.info(f'Macro F1 score: Train F1score: {train_macro_f1score:.5f}, Test dataset F1score: {test_macro_f1score:.5f}')
    logger.info(f'Micro F1 score: Train F1score: {train_micro_f1score:.5f}, Test dataset F1score: {test_micro_f1score:.5f}')
    logger.info(f"MRR score for test dataset: {test_mrr_score:.5f}")
    for k in [1,2,3,4,5,6,7,8,9,10,20,50]:
        test_hitk_score = utils.calculate_hitk(test_data, random_data, N=args.n_random_test_mrr_hk*2, k=k)
        logger.info(f"Hit@{k} for test dataset: {test_hitk_score:.8f}")
        all_evaluation_results['evaluation_hit_at_k_score'] += [test_hitk_score]

    logger.info('Scaled to 2class classification (in order to compared with other 2class models)')
    train_acc_2class = utils.calculate_acc(train_y_probs[train_data['y']!=2][:,:2], train_y_true[train_data['y']!=2])
    test_acc_2class = utils.calculate_acc(test_y_probs[test_data['y']!=2][:,:2], test_y_true[test_data['y']!=2])
    logger.info(f'Final Accuracy: Train Accuracy: {train_acc_2class:.8f}, Test dataset Accuracy: {test_acc_2class:.8f}')
    train_macro_f1score_2class = utils.calculate_f1score(train_y_probs[train_data['y']!=2][:,:2],train_y_true[train_data['y']!=2],None).mean()
    test_macro_f1score_2class = utils.calculate_f1score(test_y_probs[test_data['y']!=2][:,:2], test_y_true[test_data['y']!=2],None)[:2].mean()

    logger.info(f'Macro F1 score: Train F1score: {train_macro_f1score_2class:.8f}, Test dataset F1score: {test_macro_f1score_2class:.8f}')

    # Saves all evaluation result data for downstream analysis
    all_evaluation_results['evaluation_acc_score'] = [train_acc, test_acc]
    all_evaluation_results['evaluation_macro_f1_score'] = [train_macro_f1score, test_macro_f1score]
    all_evaluation_results['evaluation_micro_f1_score'] = [train_micro_f1score, test_micro_f1score]
    all_evaluation_results['evaluation_mrr_score'] = [test_mrr_score]
    all_evaluation_results['evaluation_y_true'] = [train_y_true, test_y_true]
    all_evaluation_results['evaluation_y_probas'] = [train_y_probs, test_y_probs]
    with open(os.path.join(args.output_folder, folder_name, 'all_evaluation_results.pkl'),'wb') as outfile:
        pickle.dump(all_evaluation_results, outfile)

    ## read all true positive pairs
    all_tp_pairs = pd.read_csv(os.path.join(args.data_dir, 'tp_pairs.txt'), sep='\t', header=0)
    all_tp_pairs_dict = {"_".join(pair):1 for pair in all_tp_pairs.to_numpy()}

    ### calculate the ranks of true positive pairs among all drug-specific disease pairs for evaluation 
    entity2id, id2entity = utils.load_index(os.path.join(args.data_dir, 'entity2freq.txt'))
    relation2id, id2relation = utils.load_index(os.path.join(args.data_dir, 'relation2freq.txt'))

    ## find all drug ids and all disease ids
    type2id, id2type = utils.load_index(os.path.join(args.data_dir, 'type2freq.txt'))
    with open(os.path.join(args.data_dir, 'entity2typeid.pkl'), 'rb') as infile:
        entity2typeid = pickle.load(infile)
    drug_type = ['biolink:Drug', 'biolink:SmallMolecule']
    drug_type_ids = [type2id[x] for x in drug_type]
    all_drug_ids = [id2entity[index] for index, typeid in enumerate(entity2typeid) if typeid in drug_type_ids]
    disease_type = ['biolink:Disease', 'biolink:PhenotypicFeature', 'biolink:BehavioralFeature', 'biolink:DiseaseOrPhenotypicFeature']
    disease_type_ids = [type2id[x] for x in disease_type]
    all_disease_ids = [id2entity[index] for index, typeid in enumerate(entity2typeid) if typeid in disease_type_ids]

    # find all drug and disease ids of true positive pairs in test data set
    test_data_pos = test_data.loc[test_data['y'] == 1,:].reset_index(drop=True)
    test_data_pos['predicted_y'] = fitModel.predict_proba(generate_X_and_y(test_data_pos, pair_emb=args.pair_emb)[0]).argmax(axis=1)
    test_data_pos.to_csv(os.path.join(args.output_folder, folder_name, 'test_data_pos.txt'), sep='\t', index=None)

    rank_res = utils.calculate_rank(test_data_pos, all_drug_ids, all_disease_ids, entity_embeddings_dict, all_tp_pairs_dict, fitModel, mode='both')
    with open(os.path.join(args.output_folder, folder_name, 'test_data_rank.pkl'), 'wb') as outfile:
        pickle.dump(rank_res, outfile)


    logger.info('##### all drug - all disease rank evaluation metrics #####')
    if list(rank_res.items())[0][1][0][0]:
        logger.info('##### without filtering other tp pairs #####')
        drug_rank_res = np.array([1 - rank_res[x][0][0][0]/rank_res[x][0][0][1] for x in rank_res]).mean()
        disease_rank_res = np.array([1 - rank_res[x][0][1][0]/rank_res[x][0][1][1] for x in rank_res]).mean()
        both_rank_res = np.array([1 - rank_res[x][0][2][0]/rank_res[x][0][2][1] for x in rank_res]).mean()
        logger.info(f"Rank Percentile: Drug Rank:{drug_rank_res:.8f}, Disease Rank:{disease_rank_res:.8f}, Both Rank:{both_rank_res:.8f}")
        drug_rank_res = np.array([rank_res[x][0][0][0] for x in rank_res]).mean()
        disease_rank_res = np.array([rank_res[x][0][1][0] for x in rank_res]).mean()
        both_rank_res = np.array([rank_res[x][0][2][0] for x in rank_res]).mean()
        logger.info(f"MR: Drug Rank:{drug_rank_res:.8f}, Disease Rank:{disease_rank_res:.8f}, Both Rank:{both_rank_res:.8f}")
        drug_rank_res = np.array([1/rank_res[x][0][0][0] for x in rank_res]).mean()
        disease_rank_res = np.array([1/rank_res[x][0][1][0] for x in rank_res]).mean()
        both_rank_res = np.array([1/rank_res[x][0][2][0] for x in rank_res]).mean()
        logger.info(f"MRR: Drug Rank:{drug_rank_res:.8f}, Disease Rank:{disease_rank_res:.8f}, Both Rank:{both_rank_res:.8f}")
        for k in [1,2,3,4,5,6,7,8,9,10,20,50,100,200,500,1000]:
            drug_rank_res = np.array([True if rank_res[x][0][0][0] <= k else False for x in rank_res]).astype(float).mean()
            disease_rank_res = np.array([True if rank_res[x][0][1][0] <= k else False for x in rank_res]).astype(float).mean()
            both_rank_res = np.array([True if rank_res[x][0][2][0] <= k else False for x in rank_res]).astype(float).mean()
            logger.info(f"Hit@{k}: Drug Rank:{drug_rank_res:.8f}, Disease Rank:{disease_rank_res:.8f}, Both Rank:{both_rank_res:.8f}")


    if list(rank_res.items())[0][1][1][0]:
        logger.info('##### filtering other tp pairs #####')
        drug_rank_res = np.array([1 - rank_res[x][1][0][0]/rank_res[x][1][0][1] for x in rank_res]).mean()
        disease_rank_res = np.array([1 - rank_res[x][1][1][0]/rank_res[x][1][1][1] for x in rank_res]).mean()
        both_rank_res = np.array([1 - rank_res[x][1][2][0]/rank_res[x][1][2][1] for x in rank_res]).mean()
        logger.info(f"Rank Percentile: Drug Rank:{drug_rank_res:.8f}, Disease Rank:{disease_rank_res:.8f}, Both Rank:{both_rank_res:.8f}")
        drug_rank_res = np.array([rank_res[x][1][0][0] for x in rank_res]).mean()
        disease_rank_res = np.array([rank_res[x][1][1][0] for x in rank_res]).mean()
        both_rank_res = np.array([rank_res[x][1][2][0] for x in rank_res]).mean()
        logger.info(f"MR: Drug Rank:{drug_rank_res:.8f}, Disease Rank:{disease_rank_res:.8f}, Both Rank:{both_rank_res:.8f}")
        drug_rank_res = np.array([1/rank_res[x][1][0][0] for x in rank_res]).mean()
        disease_rank_res = np.array([1/rank_res[x][1][1][0] for x in rank_res]).mean()
        both_rank_res = np.array([1/rank_res[x][1][2][0] for x in rank_res]).mean()
        logger.info(f"MRR: Drug Rank:{drug_rank_res:.8f}, Disease Rank:{disease_rank_res:.8f}, Both Rank:{both_rank_res:.8f}")
        for k in [1,2,3,4,5,6,7,8,9,10,20,50,100,200,500,1000]:
            drug_rank_res = np.array([True if rank_res[x][1][0][0] <= k else False for x in rank_res]).astype(float).mean()
            disease_rank_res = np.array([True if rank_res[x][1][1][0] <= k else False for x in rank_res]).astype(float).mean()
            both_rank_res = np.array([True if rank_res[x][1][2][0] <= k else False for x in rank_res]).astype(float).mean()
            logger.info(f"Hit@{k}: Drug Rank:{drug_rank_res:.8f}, Disease Rank:{disease_rank_res:.8f}, Both Rank:{both_rank_res:.8f}")
