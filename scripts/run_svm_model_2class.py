import sys
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.svm import SVC
import sklearn.metrics as met
import time
import argparse
import joblib
from sklearn.model_selection import GridSearchCV
from glob import glob
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
    parser.add_argument("--log_name", type=str, help="log file name", default="step8_svm_2class.log")
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
        data_list.append(pd.read_csv(os.path.join(args.data_dir, 'pretrain_reward_shaping_model_train_val_test_random_data_2class', f'{dataset}_pairs.txt'), sep='\t', header=0))

    ## prepare the feature vectors
    train_data = data_list[0]
    test_data = data_list[2]
    random_data = data_list[3]
    train_X, train_y = generate_X_and_y(train_data, pair_emb=args.pair_emb)
    test_X, test_y = generate_X_and_y(test_data, pair_emb=args.pair_emb)
    random_X, random_y = generate_X_and_y(random_data, pair_emb=args.pair_emb)

    ## create model folder
    folder_name = 'SVM_model_2class'
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
    SVM_model = SVC(class_weight='balanced', gamma='auto', random_state=args.seed, probability=True)
    param_grid = { 'kernel' : ['linear', 'poly', 'rbf', 'sigmoid']}
    gs_svm = GridSearchCV(estimator=SVM_model, param_grid=param_grid, cv= 10, scoring='f1_macro', return_train_score=True)
    gs_svm.fit(train_X, train_y)
    logger.info(f"The best hyper-parameter set of SVM model based on gridseaerchCV is {gs_svm.best_estimator_}")
    best_svm = gs_svm.best_estimator_
    # save grid search cv results 
    with open(os.path.join(args.output_folder, folder_name, 'grid_search_cv_results.pkl'), 'wb') as outfile:
        pickle.dump(gs_svm.cv_results_, outfile)
    fitModel = best_svm.fit(train_X, train_y)

    # # saves the model
    model_name = f'SVM_model.pt'
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

    with open(os.path.join(args.output_folder, folder_name, f'SVM_results.pkl'), 'wb') as outfile:
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

    ## Saves all evaluation result data for downstream analysis
    all_evaluation_results = dict()
    all_evaluation_results['evaluation_acc_score'] = [train_acc, test_acc]
    all_evaluation_results['evaluation_macro_f1_score'] = [train_macro_f1score, test_macro_f1score]
    all_evaluation_results['evaluation_micro_f1_score'] = [train_micro_f1score, test_micro_f1score]
    all_evaluation_results['evaluation_mrr_score'] = [test_mrr_score]
    all_evaluation_results['evaluation_y_true'] = [train_y_true, test_y_true]
    all_evaluation_results['evaluation_y_probas'] = [train_y_probs, test_y_probs]
    with open(os.path.join(args.output_folder, folder_name, 'all_evaluation_results.pkl'),'wb') as outfile:
        pickle.dump(all_evaluation_results, outfile)
