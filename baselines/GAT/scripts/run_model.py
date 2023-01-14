import pandas as pd
import torch
import os, sys
import numpy as np
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from dataset import ProcessedDataset
from model import GAT
import torch.nn.functional as F
import pickle
import argparse
import utils
import gc

def train(epoch, args, train_loader, train_batch, val_loader, val_batch, logger):

    logger.info(f"======== Epoch {epoch + 1} / {num_epochs} ========")

    model.train()

    total_loss = 0
    all_pred = torch.tensor([])
    all_y = torch.tensor([])
    for batch_idx, (n_id, adjs) in tqdm(enumerate(train_loader)):

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
        link = train_batch[batch_idx][['source','target']].apply(lambda row: [idx_map.get(row[0]),idx_map.get(row[1])], axis=1, result_type='expand').rename(columns={0: "source", 1: "target"})
        link = torch.tensor(np.array(link), dtype=torch.int).to(args.device)

        optimizer.zero_grad()
        pred_y= model(init_mat, adjs, link, n_id)
        train_loss = F.binary_cross_entropy_with_logits(pred_y, y, weights)
        if args.use_gpu:
            utils.empty_gpu_cache(args)
        all_pred = torch.cat([all_pred,torch.sigmoid(pred_y).cpu().detach()])
        all_y = torch.cat([all_y, y.cpu().detach()])
        train_loss.backward()
        if args.use_gpu:
            utils.empty_gpu_cache(args)
        optimizer.step()

        total_loss += float(train_loss.cpu().detach())

    train_loss = total_loss / len(train_loader)
    train_acc = utils.calculate_acc(all_pred, all_y, is_gnn=True)

    ## evaluate model with validation data
    model.eval()
    total_loss = 0
    all_pred = torch.tensor([])
    all_y = torch.tensor([])

    with torch.no_grad():
        for batch_idx, (n_id, adjs) in tqdm(enumerate(val_loader)):
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
            link = val_batch[batch_idx][['source','target']].apply(lambda row: [idx_map.get(row[0]),idx_map.get(row[1])], axis=1, result_type='expand').rename(columns={0: "source", 1: "target"})
            link = torch.tensor(np.array(link), dtype=torch.int).to(args.device)

            pred_y = model(init_mat, adjs, link, n_id).detach()
            if args.use_gpu:
                utils.empty_gpu_cache(args)
            val_loss = F.binary_cross_entropy_with_logits(pred_y, y, weights)
            all_pred = torch.cat([all_pred,torch.sigmoid(pred_y).cpu().detach()])
            all_y = torch.cat([all_y, y.cpu().detach()])

            total_loss += float(val_loss.cpu().detach())

        val_loss = total_loss / len(val_loader)
        val_acc = utils.calculate_acc(all_pred,all_y, is_gnn=True)

    logger.info(f"Epoch Stat: Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.5f}, Val Loss: {val_loss:.5f}, Val Acc: {val_acc:.5f}")

    return [train_loss, train_acc, val_loss, val_acc]


def evaluate(args, loader, batch_data, calculate_metric=True):
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for batch_idx, (n_id, adjs) in tqdm(enumerate(loader)):
            n_id = n_id[0].to(args.device)
            adjs = [(adj[0][0],(int(adj[1][0]),int(adj[1][1]))) for adj in adjs]
            link = batch_data[batch_idx][['source','target']].apply(lambda row: [idx_map.get(row[0]),idx_map.get(row[1])], axis=1, result_type='expand').rename(columns={0: "source", 1: "target"})
            link = torch.tensor(np.array(link), dtype=torch.long).to(args.device)

            pred = torch.sigmoid(model(init_mat, adjs, link, n_id)).detach().cpu().numpy()
            if args.use_gpu:
                utils.empty_gpu_cache(args)
            label = np.array(batch_data[batch_idx]['y'])

            predictions.append(pred)
            labels.append(label)

    probas = np.hstack(predictions)
    labels = np.hstack(labels)

    if calculate_metric is True:

        ## calculate accuracy
        acc = utils.calculate_acc(probas,labels, is_gnn=True)
        ## calculate macro F1 score
        macro_f1score = utils.calculate_f1score(probas,labels,'macro', is_gnn=True)
        ## calculate micro F1 score
        micro_f1score = utils.calculate_f1score(probas,labels,'micro', is_gnn=True)

        return [acc, macro_f1score, micro_f1score, labels, probas]

    else:
        return [None, None, None, labels, probas]

####################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main script for running model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--log_dir", type=str, help="The path to logfile folder", default="~/DTD_RL_model/log_folder")
    parser.add_argument("--log_name", type=str, help="log file name", default="additional_GAT_run.log")
    parser.add_argument("--data_dir", type=str, help="The path to data folder", default="~/DTD_RL_model/data")
    parser.add_argument("--processed_data_dir", type=str, help="The path to data folder", default="~/DTD_RL_model/data/GAT_processed_data")
    parser.add_argument('--gpu', type=int, help='gpu device (default: 0)', default=0)
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU to train model", default=False)
    parser.add_argument('--seed', type=int, help='Random seed (default: 1023)', default=1023)
    parser.add_argument("--learning_ratio", type=float, help="Learning ratio", default=0.001)
    parser.add_argument("--init_emb_size", type=int, help="Initial embedding", default=100)
    parser.add_argument("--num_samples", type=str, help="Number of sample in different layers (The length should match 'num_layers')", default="None")
    parser.add_argument("--use_known_embedding", action="store_true", help="Use known inital embeeding", default=False)
    parser.add_argument("--known_embedding", type=str, help="The path to known embeddings", default="~/DTD_RL_model/data/embedding_biobert_namecat.pkl")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs to train model", default=50)
    parser.add_argument("--n_random_test_mrr_hk", type=int, help="Number of random pairs assigned to each TP drug in test set for MRR and H@K", default=500)
    parser.add_argument("--emb_size", type=int, help="Embedding vertor dimension", default=128)
    parser.add_argument("--batch_size", type=int, help="Batch size of training data", default=500)
    parser.add_argument("--num_layers", type=int, help="Number of GNN layers to train model", default=2)
    parser.add_argument("--patience", type=int, help="Number of epochs with no improvement after which learning rate will be reduced", default=10)
    parser.add_argument("--factor", type=float, help="The factor for learning rate to be reduced", default=0.1)
    parser.add_argument("--early_stop_n", type=int, help="Early stop if validation loss doesn't further decrease after n step", default=50)
    parser.add_argument("--num_head", type=int, help="Number of head in GAT model", default=4)
    parser.add_argument("--dropout_p", type=float, help="Drop out proportion", default=0)
    parser.add_argument("--output_folder", type=str, help="The path to output folder", default="~/work/explainable_DTD_model/results")
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir,args.log_name))
    logger.info(args)
    utils.set_random_seed(args.seed)

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    embedding_size = args.emb_size
    num_layers = args.num_layers
    dropout_p = args.dropout_p
    lr = args.learning_ratio
    num_head = args.num_head
    init_emb_size = args.init_emb_size
    patience = args.patience
    factor = args.factor
    early_stop_n = args.early_stop_n
    n_random_test_mrr_hk = args.n_random_test_mrr_hk
    processed_data_dir = args.processed_data_dir
    if args.num_samples == "None":
        num_samples = None
    else:
        num_samples = eval(args.num_samples)

    if num_samples is not None:
        assert len(num_samples) == num_layers

    if args.use_gpu and torch.cuda.is_available():
        use_gpu = True
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.set_device(args.gpu)
    elif args.use_gpu:
        logger.warning('No GPU is detected in this computer. Use CPU instead.')
        use_gpu = False
        device = 'cpu'
    else:
        use_gpu = False
        device = 'cpu'
    args.use_gpu = use_gpu
    args.device = device

    ## genreate a processed data folder
    processdata_path = os.path.join(processed_data_dir, f'GAT_ProcessedDataset_initemb{init_emb_size}_batch{batch_size}_layer{num_layers}')
    if not os.path.isdir(processdata_path):
        os.makedirs(processdata_path)

    logger.info('Start pre-processing data')
    dataset = ProcessedDataset(root=processdata_path, args=args, batch_size=batch_size, layers=num_layers, dim=init_emb_size, num_samples=num_samples, logger=logger)
    logger.info('Pre-processing data completed')
    data = dataset.get_dataset()
    idx_map, id_to_type, typeid = dataset.get_mapfiles()
    train_batch, val_batch, test_batch, random_batch = dataset.get_train_val_test_random()
    train_loader = dataset.get_train_loader()
    val_loader = dataset.get_val_loader()
    test_loader = dataset.get_test_loader()
    random_loader = dataset.get_random_loader()
    init_emb = data.feat

    model = GAT(args, init_emb_size, embedding_size, 1, num_layers=num_layers, dropout_p=dropout_p, num_head = num_head)
    model = model.to(args.device)
    folder_name = f'GAT_batchsize{batch_size}_initemb{init_emb_size}_embeddingsize{embedding_size}_layers{num_layers}_numhead{num_head}_lr{lr}_epoch{num_epochs:05d}_patience{patience}_factor{factor}'
    if not os.path.exists(os.path.join(args.output_folder, folder_name)):
        os.makedirs(os.path.join(args.output_folder, folder_name))
    writer = SummaryWriter(log_dir=os.path.join(args.output_folder, folder_name, 'tensorboard_runs'))

    all_sorted_indexes = torch.hstack([init_emb[key][1] for key,value in init_emb.items()]).sort().indices
    all_init_mats = [init_emb[key][0] for key,value in init_emb.items()]
    init_mat = torch.vstack([mat for _, mat in enumerate(all_init_mats)])[all_sorted_indexes]
    init_mat = init_mat.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, threshold=0.0001, threshold_mode='rel')

    all_train_loss = []
    all_val_loss = []
    all_train_acc = []
    all_val_acc = []

    current_min_val_loss = float('inf')
    model_state_dict = None
    count = 0
    logger.info('Start training model')
    for epoch in trange(num_epochs):
        if count > early_stop_n:
            break
        train_loss, train_acc, val_loss, val_acc = train(epoch, args, train_loader, train_batch, val_loader, val_batch, logger)
        scheduler.step(val_loss)
        all_train_loss += [train_loss]
        all_val_loss += [val_loss]
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        all_train_acc += [train_acc]
        all_val_acc += [val_acc]
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        count += 1
        if val_loss < current_min_val_loss:
            count = 0
            current_min_val_loss = val_loss
            model_state_dict = model.state_dict()
            model_name = f'model_epoch{epoch+1:05d}_val_loss{current_min_val_loss:.5f}.pt'

    writer.close()
    ## Saves model and weights
    torch.save({'model_state_dict': model_state_dict}, os.path.join(args.output_folder, folder_name, model_name))

    ## Saves data for plotting graph
    epoches = list(range(1,num_epochs+1))
    plotdata_loss = pd.DataFrame(list(zip(epoches,all_train_loss,['train_loss']*num_epochs)) + list(zip(epoches,all_val_loss,['val_loss']*num_epochs)), columns=['epoch', 'loss', 'type'])
    plotdata_acc = pd.DataFrame(list(zip(epoches,all_train_acc,['train_acc']*num_epochs)) + list(zip(epoches,all_val_acc,['val_acc']*num_epochs)), columns=['epoch', 'acc', 'type'])
    acc_loss_plotdata = [epoches, plotdata_loss, plotdata_acc]

    with open(os.path.join(args.output_folder, folder_name, 'acc_loss_plotdata.pkl'), 'wb') as file_out:
        pickle.dump(acc_loss_plotdata, file_out)

    logger.info('Load in the best model')
    model = GAT(args, init_emb_size, embedding_size, 1, num_layers=num_layers, dropout_p=dropout_p, num_head = num_head)
    model = model.to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.output_folder, folder_name, model_name))['model_state_dict'])

    logger.info("")
    logger.info('#### Evaluate best model ####')
    train_acc, train_macro_f1score, train_micro_f1score, train_y_true, train_y_probs = evaluate(args, train_loader, train_batch)
    # _, _, _, val_y_true, val_y_probs = evaluate(args, val_loader, val_batch, calculate_metric=False)
    _, _, _, test_y_true, test_y_probs = evaluate(args, test_loader, test_batch, calculate_metric=False)
    val_test_y_true = test_y_true
    # val_test_y_true = np.hstack([val_y_true, test_y_true])
    val_test_y_probs = test_y_probs
    # val_test_y_probs = np.hstack([val_y_probs, test_y_probs])
    ## calculate accuracy
    val_test_acc = utils.calculate_acc(val_test_y_probs,val_test_y_true, is_gnn=True)
    ## calculate macro F1 score
    val_test_macro_f1score = utils.calculate_f1score(val_test_y_probs,val_test_y_true,'macro', is_gnn=True)
    ## calculate micro F1 score
    val_test_micro_f1score = utils.calculate_f1score(val_test_y_probs,val_test_y_true,'micro', is_gnn=True)

    train_data = pd.concat(train_batch)
    train_data['prob'] = train_y_probs
    train_data['pred_y'] = (train_y_probs > 0.5) * 1
    # val_data = pd.concat(val_batch)
    test_data = pd.concat(test_batch)
    val_test_data = test_data
    # val_test_data = pd.concat([val_data, test_data])
    val_test_data['prob'] = val_test_y_probs
    val_test_data['pred_y'] = (val_test_y_probs > 0.5) * 1
    _, _, _, random_y_true, random_y_probs = evaluate(args, random_loader, random_batch, calculate_metric=False)
    random_data = pd.concat(random_batch)
    random_data['prob'] = random_y_probs
    random_data['pred_y'] = (random_y_probs > 0.5) * 1

    all_evaluation_results = dict()
    all_evaluation_results['evaluation_hit_at_k_score'] = []
    val_test_mrr_score = utils.calculate_mrr(val_test_data, random_data, N=n_random_test_mrr_hk*2)
    logger.info(f'Accuracy: Train Accuracy: {train_acc:.5f}, Test dataset Accuracy: {val_test_acc:.5f}')
    logger.info(f'Macro F1 score: Train F1score: {train_macro_f1score:.5f}, Test dataset F1score: {val_test_macro_f1score:.5f}')
    logger.info(f'Micro F1 score: Train F1score: {train_micro_f1score:.5f}, Test dataset F1score: {val_test_micro_f1score:.5f}')
    logger.info(f"MRR score for test data: {val_test_mrr_score:.5f}")
    for k in [1,2,3,4,5,6,7,8,9,10,20,50]:
        val_test_hitk_score = utils.calculate_hitk(val_test_data, random_data, N=n_random_test_mrr_hk*2, k=k)
        logger.info(f"Hit@{k} for test data: {val_test_hitk_score:.8f}")
        all_evaluation_results['evaluation_hit_at_k_score'] += [val_test_hitk_score]

    ## Saves all evaluation result data for downstream analysis
    all_evaluation_results = dict()
    all_evaluation_results['evaluation_acc_score'] = [train_acc, val_test_acc]
    all_evaluation_results['evaluation_macro_f1_score'] = [train_macro_f1score, val_test_macro_f1score]
    all_evaluation_results['evaluation_micro_f1_score'] = [train_micro_f1score, val_test_micro_f1score]
    all_evaluation_results['evaluation_mrr_score'] = [val_test_mrr_score]
    all_evaluation_results['evaluation_y_true'] = [train_y_true, val_test_y_true]
    all_evaluation_results['evaluation_y_probas'] = [train_y_probs, val_test_y_probs]
    with open(os.path.join(args.output_folder, folder_name, 'all_evaluation_results.pkl'),'wb') as outfile:
        pickle.dump(all_evaluation_results, outfile)

    print('#### Program Summary ####')
    if use_gpu is True:
        for index in range(torch.cuda.device_count()):
            print(f'Max memory used by tensors = {torch.cuda.max_memory_allocated(index)} bytes for GPU:{index}')
            print(f'Max memory managed by caching allocator = {torch.cuda.max_memory_reserved(index)} bytes for GPU:{index}')
        gc.collect()
        torch.cuda.empty_cache()
