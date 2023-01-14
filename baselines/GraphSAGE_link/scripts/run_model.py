import pandas as pd
import torch
import os
import numpy as np
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from dataset import ProcessedDataset
from model import GraphSage
import torch.nn.functional as F
import pickle
import argparse
import utils
import time
import gc
import copy
import sklearn.metrics as met
import scipy as sci
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def train(epoch, num_epochs, train_loader, train_batch, val_loader, val_batch, args):
    print("")
    print(f"======== Epoch {epoch + 1} / {num_epochs} ========")
    print('Training...')
    model.train()
    t0 = time.time()

    total_loss = 0
    all_pred = torch.tensor([])
    all_y = torch.tensor([])
    for batch_idx, (n_id, adjs) in enumerate(train_loader):
        
        batch_t0 = time.time()
        n_id = n_id[0].to(args.device)
        adjs = [(adj[0][0],(int(adj[1][0]),int(adj[1][1]))) for adj in adjs]
        y = torch.tensor(train_batch[batch_idx]['y'], dtype=torch.float).to(args.device)
        # deal with inblance class with weights
        pos = len(torch.where(y == 1)[0])
        neg = len(torch.where(y == 0)[0])
        n_sample = neg + pos
        weights = torch.zeros(n_sample, dtype=torch.float)
        if neg > pos:
            weights[torch.where(y == 1)[0]] = neg/pos
            weights[torch.where(y == 0)[0]] = 1
        elif pos > neg:
            weights[torch.where(y == 1)[0]] = 1
            weights[torch.where(y == 0)[0]] = pos/neg
        else:
            weights[torch.where(y == 1)[0]] = 1
            weights[torch.where(y == 0)[0]] = 1
        weights = weights.to(args.device)
        link = train_batch[batch_idx][['source','target']].apply(lambda row: [entity2id.get(row[0]) - 1, entity2id.get(row[1]) - 1], axis=1, result_type='expand').rename(columns={0: "source", 1: "target"})
        link = torch.tensor(np.array(link), dtype=torch.int).to(args.device)
        x_n_id = init_emb[n_id]

        optimizer.zero_grad()
        pred_y= model(x_n_id, adjs, link, n_id)
        train_loss = F.binary_cross_entropy_with_logits(pred_y, y, weights)
        all_pred = torch.cat([all_pred,torch.sigmoid(pred_y).cpu().detach()])
        all_y = torch.cat([all_y, y.cpu().detach()])
        train_loss.backward()
        optimizer.step()           
        
        total_loss += float(train_loss.detach())
        
        if (batch_idx+1) % args.print_every == 0:
            elapsed = utils.format_time(time.time() - batch_t0)
            args.logger.info(f"Batch {batch_idx+1} of {len(train_loader)}. This batch costs around: {elapsed}.")

    ## calculate total loss
    train_loss = total_loss / len(train_loader)
    ## calculate accuracy
    train_acc = utils.calculate_acc(all_pred, all_y)
    ## calculate macro F1 score
    train_macro_f1score = utils.calculate_f1score(all_pred, all_y, mode='macro')

    ## evaluate model with validation data
    model.eval()
    total_loss = 0
    all_pred = torch.tensor([])
    all_y = torch.tensor([])
    with torch.no_grad():
        for batch_idx, (n_id, adjs) in enumerate(val_loader):

            n_id = n_id[0].to(args.device)
            adjs = [(adj[0][0],(int(adj[1][0]),int(adj[1][1]))) for adj in adjs]
            y = torch.tensor(val_batch[batch_idx]['y'], dtype=torch.float).to(args.device)
            # deal with inblance class with weights
            pos = len(torch.where(y == 1)[0])
            neg = len(torch.where(y == 0)[0])
            n_sample = neg + pos
            weights = torch.zeros(n_sample, dtype=torch.float)
            if neg > pos:
                weights[torch.where(y == 1)[0]] = neg/pos
                weights[torch.where(y == 0)[0]] = 1
            elif pos > neg:
                weights[torch.where(y == 1)[0]] = 1
                weights[torch.where(y == 0)[0]] = pos/neg
            else:
                weights[torch.where(y == 1)[0]] = 1
                weights[torch.where(y == 0)[0]] = 1
            weights = weights.to(args.device)
            link = val_batch[batch_idx][['source','target']].apply(lambda row: [entity2id.get(row[0]) - 1, entity2id.get(row[1]) - 1], axis=1, result_type='expand').rename(columns={0: "source", 1: "target"})
            link = torch.tensor(np.array(link), dtype=torch.int).to(args.device)
            x_n_id = init_emb[n_id]

            pred_y = model(x_n_id, adjs, link, n_id).detach()
            val_loss = F.binary_cross_entropy_with_logits(pred_y, y, weights)
            all_pred = torch.cat([all_pred,torch.sigmoid(pred_y).cpu().detach()])
            all_y = torch.cat([all_y, y.cpu().detach()])

            total_loss += float(val_loss.detach())
        
        ## calculate total loss
        val_loss = total_loss / len(val_loader)
        ## calculate accuracy
        val_acc = utils.calculate_acc(all_pred,all_y)
        ## calculate macro F1 score
        val_macro_f1score = utils.calculate_f1score(all_pred, all_y, mode='macro')

    args.logger.info(f"Epoch Stat: Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.5f}, Train Macro F1 score {train_macro_f1score:5f}, Val Loss: {val_loss:.5f}, Val Acc: {val_acc:.5f}, Val Macro F1 score {val_macro_f1score:5f}")
    training_time = utils.format_time(time.time() - t0)
    args.logger.info(f"The total running time of this epoch: {training_time}")

    return [train_loss, train_acc, train_macro_f1score, val_loss, val_acc, val_macro_f1score]


def evaluate(model, loaders, batch_data_list, args, calculate_metric=True): 
    model.eval()

    predictions = []
    y_true = []

    with torch.no_grad():
        for index, loader in enumerate(loaders):
            for batch_idx, (n_id, adjs) in enumerate(loader):
                n_id = n_id[0].to(args.device)
                adjs = [(adj[0][0],(int(adj[1][0]),int(adj[1][1]))) for adj in adjs]
                link = batch_data_list[index][batch_idx][['source','target']].apply(lambda row: [entity2id.get(row[0]) - 1,entity2id.get(row[1]) - 1], axis=1, result_type='expand').rename(columns={0: "source", 1: "target"})
                link = torch.tensor(np.array(link), dtype=torch.long).to(args.device)
                x_n_id = init_emb[n_id]

                pred = model.predict1(x_n_id, adjs, link, n_id).numpy()
                predictions.append(pred)
                y_true.append(np.array(batch_data_list[index][batch_idx]['y']))

    probas = np.concatenate(predictions)
    y_true = np.hstack(y_true)

    if calculate_metric is True:
        
        ## calculate accuracy
        acc = utils.calculate_acc(probas[:,1], y_true)
        ## calculate macro F1 score
        macro_f1score = utils.calculate_f1score(probas[:,1], y_true, mode='macro')
        ## calculate micro F1 score
        micro_f1score = utils.calculate_f1score(probas[:,1], y_true, mode='micro')

        
        return [acc, macro_f1score, micro_f1score, y_true, probas]
    
    else:
        
        return [None, None, None, y_true, probas]

####################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main script for running model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ## general parameters
    parser.add_argument("--data_path", type=str, help="Data Forlder", default="../data")
    parser.add_argument("--result_folder", type=str, help="The path of result folder", default="~/results")

    ## model parameters
    parser.add_argument("--use_known_embedding", action="store_true", help="Use known inital embeeding", default=False)
    parser.add_argument("--init_emb_size", type=int, help="Model initial embedding size if no known embedding is specified", default=100)
    parser.add_argument("--init_emb_file", type=str, help="Initial embedding", default="../data/embedding_biobert_namecat.pkl")
    parser.add_argument("--hidden_emb_size", type=int, help="Model hidden embedding size", default=256)
    parser.add_argument("--num_layers", type=int, help="Number of GNN layers to train model", default=3)
    parser.add_argument("--layer_size", type=int, nargs='*', help="Sample size for each layer", default=[96, 96])
    parser.add_argument("--dropout_p", type=float, help="Drop out proportion", default=0.2)

    ## other parameters
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU to train model", default=False)
    parser.add_argument("--use_multiple_gpu", action="store_true", help="Use all GPUs on computer to train model", default=False)
    parser.add_argument("--seed", type=float, help="Manually set initial seed for pytorch", default=1020)
    parser.add_argument("--learning_ratio", type=float, help="Learning ratio", default=0.001)
    parser.add_argument('--gpu', type=int, help='gpu device (default: 0) if use_multiple_gpu is False', default=0)
    parser.add_argument("--num_epochs", type=int, help="Number of epochs to train model", default=100)
    parser.add_argument("--batch_size", type=int, help="Batch size of training data", default=512)
    parser.add_argument("--patience", type=int, help="Number of epochs with no improvement after which learning rate will be reduced", default=10)
    parser.add_argument("--factor", type=float, help="The factor for learning rate to be reduced", default=0.1)
    parser.add_argument("--early_stop_n", type=int, help="Early stop if validation loss doesn't further decrease after n step", default=50)
    parser.add_argument("--n_random_test_mrr_hk", type=int, help="Number of random pairs assigned to each TP drug in test set for MRR and H@K", default=500)
    parser.add_argument("--print_every", type=int, help="How often to print training batch elapsed time", default=10)

    args = parser.parse_args()
    start_time = time.time()

    if args.use_known_embedding:
        with open(args.init_emb_file,'rb') as infile:
            args.known_int_emb_dict = pickle.load(infile)
    else:
        args.known_int_emb_dict = None
    
    utils.set_random_seed(args.seed)
    args.logger = utils.get_logger()

    if args.use_gpu and torch.cuda.is_available():
        use_gpu = True
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.set_device(args.gpu)
    elif args.use_gpu:
        args.logger.info('No GPU is detected in this computer. Use CPU instead.')
        use_gpu = False
        device = 'cpu'
    else:
        use_gpu = False
        device = 'cpu'
    args.use_gpu = use_gpu
    args.device = device

    if args.use_known_embedding:
        processdata_path = os.path.join(args.data_path, f'ProcessedDataset_pretrained_initemb_batch{args.batch_size}_layer{args.num_layers}')
    else:
        processdata_path = os.path.join(args.data_path, f'ProcessedDataset_initemb{args.init_emb_size}_batch{args.batch_size}_layer{args.num_layers}')
    args.logger.info('Start pre-processing data')
    dataset = ProcessedDataset(root=processdata_path, args=args)
    args.logger.info('Pre-processing data completed')
    data = dataset.get_dataset()
    entity2id, id2entity = utils.load_index(os.path.join(args.data_path, 'entity2freq.txt'))
    train_batch, val_batch, test_batch, random_batch = dataset.get_train_val_test_random()
    train_loader = dataset.get_train_loader()
    val_loader = dataset.get_val_loader()
    test_loader = dataset.get_test_loader()
    random_loader = dataset.get_random_loader()
    init_emb = data.feat
    if args.use_known_embedding:
        args.logger.info(f"Use pretrained embeddings as initial embeddings and so set the 'init_emb_size' equal to {init_emb.shape[1]}")
        args.init_emb_size = init_emb.shape[1]

    model = GraphSage(args.init_emb_size, args.hidden_emb_size, 1, args=args)
    if args.use_known_embedding:
        folder_name = f'batchsize{args.batch_size}_pretrained_initemb_hiddemb{args.hidden_emb_size}_layers{args.num_layers}_lr{args.learning_ratio}_epoch{args.num_epochs:05d}_patience{args.patience}_factor{args.factor}'
    else:
        folder_name = f'batchsize{args.batch_size}_initemb{args.init_emb_size}_hiddemb{args.hidden_emb_size}_layers{args.num_layers}_lr{args.learning_ratio}_epoch{args.num_epochs:05d}_patience{args.patience}_factor{args.factor}'
    try:
        os.mkdir(os.path.join(args.result_folder, folder_name))
    except:
        pass
    writer = SummaryWriter(log_dir=os.path.join(args.result_folder, folder_name, 'tensorboard_runs'))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_ratio)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience, threshold=0.00001, threshold_mode='rel')
    all_train_loss = []
    all_val_loss = []
    all_train_acc = []
    all_val_acc = []
    all_train_macro_f1score = []
    all_val_macro_f1score = []
    
    current_min_val_loss = float('inf')
    model_state_dict = None
    count = 0
    args.logger.info('Start training model')
    for epoch in trange(args.num_epochs):
        if count > args.early_stop_n:
            break
        train_loss, train_acc, train_macro_f1score, val_loss, val_acc, val_macro_f1score = train(epoch, args.num_epochs, train_loader, train_batch, val_loader, val_batch, args=args)
        scheduler.step(val_loss)
        all_train_loss += [train_loss]
        all_val_loss += [val_loss]
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        all_train_acc += [train_acc]
        all_val_acc += [val_acc]
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        all_train_macro_f1score += [train_macro_f1score]
        all_val_macro_f1score += [val_macro_f1score]
        writer.add_scalars('Macro_F1_score', {'train': train_macro_f1score, 'val': val_macro_f1score}, epoch)     
        count += 1
        if val_loss < current_min_val_loss:
            count = 0
            current_min_val_loss = val_loss
            model_state_dict = copy.deepcopy(model.state_dict())
            model_name = f'GraphSage_link_batchsize{args.batch_size}_initemb{args.init_emb_size}_hiddemb{args.hidden_emb_size}_layers{args.num_layers}_lr{args.learning_ratio}_epoch{epoch:05d}_patience{args.patience}_factor{args.factor}.pt'   

    writer.close()
    ## Saves model and weights
    torch.save({'model_state_dict': model_state_dict}, os.path.join(args.result_folder, folder_name, model_name))

    print("")
    print('#### Load in the best model', flush=True)
    model = GraphSage(args.init_emb_size, args.hidden_emb_size, 1, args=args)
    model.load_state_dict(torch.load(os.path.join(args.result_folder, folder_name, model_name))['model_state_dict'])
    model.eval()

    print("")
    print('#### Evaluate best model ####', flush=True)
    train_acc, train_macro_f1score, train_micro_f1score, train_y_true, train_y_probs = evaluate(model, [train_loader], [train_batch], args=args, calculate_metric=True)
    val_test_acc, val_test_macro_f1score, val_test_micro_f1score, val_test_y_true, val_test_y_probs = evaluate(model, [test_loader], [test_batch], args=args, calculate_metric=True)
    # val_test_acc, val_test_macro_f1score, val_test_micro_f1score, val_test_y_true, val_test_y_probs = evaluate(model, [val_loader,test_loader], [val_batch,test_batch], args=args, calculate_metric=True)

    val_test_data = pd.concat(test_batch)
    # val_test_data = pd.concat(val_batch + test_batch)
    val_test_data['prob'] = val_test_y_probs[:,1]
    val_test_data['pred_y'] = np.argmax(val_test_y_probs, axis=1)

    _, _, _, random_y_true, random_y_probs = evaluate(model, [random_loader], [random_batch], args=args, calculate_metric=False)
    random_data = pd.concat(random_batch)
    random_data['prob'] = random_y_probs[:,1]
    random_data['pred_y'] = np.argmax(random_y_probs, axis=1)


    all_evaluation_results = dict()
    all_evaluation_results['evaluation_hit_at_k_score'] = []
    val_test_mrr_score = utils.calculate_mrr(val_test_data, random_data, N=args.n_random_test_mrr_hk*2)
    args.logger.info(f'Accuracy: Train Accuracy: {train_acc:.5f}, Test dataset Accuracy: {val_test_acc:.5f}')
    args.logger.info(f'Macro F1 score: Train F1score: {train_macro_f1score:.5f}, Test dataset F1score: {val_test_macro_f1score:.5f}')
    args.logger.info(f'Micro F1 score: Train F1score: {train_micro_f1score:.5f}, Test dataset F1score: {val_test_micro_f1score:.5f}')
    args.logger.info(f"MRR score for test data: {val_test_mrr_score:.5f}")
    for k in [1,2,3,4,5,6,7,8,9,10,20,50]:
        val_test_hitk_score = utils.calculate_hitk(val_test_data, random_data, N=args.n_random_test_mrr_hk*2, k=k)
        args.logger.info(f"Hit@{k} for test data: {val_test_hitk_score:.8f}")
        all_evaluation_results['evaluation_hit_at_k_score'] += [val_test_hitk_score]
    
    ## Saves all evaluation result data for downstream analysis
    all_evaluation_results = dict()
    all_evaluation_results['evaluation_acc_score'] = [train_acc, val_test_acc]
    all_evaluation_results['evaluation_macro_f1_score'] = [train_macro_f1score, val_test_macro_f1score]
    all_evaluation_results['evaluation_micro_f1_score'] = [train_micro_f1score, val_test_micro_f1score]
    all_evaluation_results['evaluation_mrr_score'] = [val_test_mrr_score]    
    all_evaluation_results['evaluation_y_true'] = [train_y_true, val_test_y_true]
    all_evaluation_results['evaluation_y_probas'] = [train_y_probs, val_test_y_probs]
    with open(os.path.join(args.result_folder, folder_name, 'all_evaluation_results.pkl'),'wb') as outfile:
        pickle.dump(all_evaluation_results, outfile)

    ## read all true positive pairs
    all_tp_pairs = pd.read_csv(os.path.join(args.data_path, 'tp_pairs.txt'), sep='\t', header=0)
    all_tp_pairs_dict = {"_".join(pair):1 for pair in all_tp_pairs.to_numpy()}

    ### calculate the ranks of true positive pairs among all drug-specific disease pairs for evaluation
    ## find all drug ids and all disease ids
    type2id, id2type = utils.load_index(os.path.join(args.data_path, 'type2freq.txt'))
    with open(os.path.join(args.data_path, 'entity2typeid.pkl'), 'rb') as infile:
        entity2typeid = pickle.load(infile)
    drug_type = ['biolink:Drug', 'biolink:SmallMolecule']
    drug_type_ids = [type2id[x] for x in drug_type]
    all_drug_ids = [id2entity[index] for index, typeid in enumerate(entity2typeid) if typeid in drug_type_ids]
    disease_type = ['biolink:Disease', 'biolink:PhenotypicFeature', 'biolink:BehavioralFeature', 'biolink:DiseaseOrPhenotypicFeature']
    disease_type_ids = [type2id[x] for x in disease_type]
    all_disease_ids = [id2entity[index] for index, typeid in enumerate(entity2typeid) if typeid in disease_type_ids]
    args.entity2id = entity2id

    ## find all drug and disease ids of true positive pairs in test data set
    val_test_data_pos = val_test_data.loc[val_test_data['y'] == 1,:].reset_index(drop=True)
    val_test_data_pos.to_csv(os.path.join(args.result_folder, folder_name, 'val_test_data_pos.txt'), sep='\t', index=None)

    rank_res = utils.calculate_rank(val_test_data_pos, all_drug_ids, all_disease_ids, all_tp_pairs_dict, data, model, args=args, mode='both')
    with open(os.path.join(args.result_folder, folder_name, 'val_test_data_rank.pkl'), 'wb') as outfile:
        pickle.dump(rank_res, outfile)


    args.logger.info('##### all drug - all disease rank evaluation metrics #####')
    if list(rank_res.items())[0][1][0][0]:
        args.logger.info('##### without filtering other tp pairs #####')
        drug_rank_res = np.array([1 - rank_res[x][0][0][0]/rank_res[x][0][0][1] for x in rank_res]).mean()
        disease_rank_res = np.array([1 - rank_res[x][0][1][0]/rank_res[x][0][1][1] for x in rank_res]).mean()
        both_rank_res = np.array([1 - rank_res[x][0][2][0]/rank_res[x][0][2][1] for x in rank_res]).mean()
        args.logger.info(f"Rank Percentile: Drug Rank:{drug_rank_res:.8f}, Disease Rank:{disease_rank_res:.8f}, Both Rank:{both_rank_res:.8f}")
        drug_rank_res = np.array([rank_res[x][0][0][0] for x in rank_res]).mean()
        disease_rank_res = np.array([rank_res[x][0][1][0] for x in rank_res]).mean()
        both_rank_res = np.array([rank_res[x][0][2][0] for x in rank_res]).mean()
        args.logger.info(f"MR: Drug Rank:{drug_rank_res:.8f}, Disease Rank:{disease_rank_res:.8f}, Both Rank:{both_rank_res:.8f}")
        drug_rank_res = np.array([1/rank_res[x][0][0][0] for x in rank_res]).mean()
        disease_rank_res = np.array([1/rank_res[x][0][1][0] for x in rank_res]).mean()
        both_rank_res = np.array([1/rank_res[x][0][2][0] for x in rank_res]).mean()
        args.logger.info(f"MRR: Drug Rank:{drug_rank_res:.8f}, Disease Rank:{disease_rank_res:.8f}, Both Rank:{both_rank_res:.8f}")
        for k in [1,2,3,4,5,6,7,8,9,10,20,50,100,200,500,1000]:
            drug_rank_res = np.array([True if rank_res[x][0][0][0] <= k else False for x in rank_res]).astype(float).mean()
            disease_rank_res = np.array([True if rank_res[x][0][1][0] <= k else False for x in rank_res]).astype(float).mean()
            both_rank_res = np.array([True if rank_res[x][0][2][0] <= k else False for x in rank_res]).astype(float).mean()
            args.logger.info(f"Hit@{k}: Drug Rank:{drug_rank_res:.8f}, Disease Rank:{disease_rank_res:.8f}, Both Rank:{both_rank_res:.8f}")


    if list(rank_res.items())[0][1][1][0]:
        args.logger.info('##### filtering other tp pairs #####')
        drug_rank_res = np.array([1 - rank_res[x][1][0][0]/rank_res[x][1][0][1] for x in rank_res]).mean()
        disease_rank_res = np.array([1 - rank_res[x][1][1][0]/rank_res[x][1][1][1] for x in rank_res]).mean()
        both_rank_res = np.array([1 - rank_res[x][1][2][0]/rank_res[x][1][2][1] for x in rank_res]).mean()
        args.logger.info(f"Rank Percentile: Drug Rank:{drug_rank_res:.8f}, Disease Rank:{disease_rank_res:.8f}, Both Rank:{both_rank_res:.8f}")
        drug_rank_res = np.array([rank_res[x][1][0][0] for x in rank_res]).mean()
        disease_rank_res = np.array([rank_res[x][1][1][0] for x in rank_res]).mean()
        both_rank_res = np.array([rank_res[x][1][2][0] for x in rank_res]).mean()
        args.logger.info(f"MR: Drug Rank:{drug_rank_res:.8f}, Disease Rank:{disease_rank_res:.8f}, Both Rank:{both_rank_res:.8f}")
        drug_rank_res = np.array([1/rank_res[x][1][0][0] for x in rank_res]).mean()
        disease_rank_res = np.array([1/rank_res[x][1][1][0] for x in rank_res]).mean()
        both_rank_res = np.array([1/rank_res[x][1][2][0] for x in rank_res]).mean()
        args.logger.info(f"MRR: Drug Rank:{drug_rank_res:.8f}, Disease Rank:{disease_rank_res:.8f}, Both Rank:{both_rank_res:.8f}")
        for k in [1,2,3,4,5,6,7,8,9,10,20,50,100,200,500,1000]:
            drug_rank_res = np.array([True if rank_res[x][1][0][0] <= k else False for x in rank_res]).astype(float).mean()
            disease_rank_res = np.array([True if rank_res[x][1][1][0] <= k else False for x in rank_res]).astype(float).mean()
            both_rank_res = np.array([True if rank_res[x][1][2][0] <= k else False for x in rank_res]).astype(float).mean()
            args.logger.info(f"Hit@{k}: Drug Rank:{drug_rank_res:.8f}, Disease Rank:{disease_rank_res:.8f}, Both Rank:{both_rank_res:.8f}")
