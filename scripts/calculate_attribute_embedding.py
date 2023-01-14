import os
import argparse
import json
import time
import pickle
import utils
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from sklearn.decomposition import PCA


def get_bert_embedding(texts, tokenizer, model, device):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = inputs.to(device)
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    return embeddings.detach().to("cpu").numpy()


def load_data(args):
    id2index = {}
    texts = []

    with open(args.data_dir, "r") as f:
        line = f.readline()
        line = f.readline()
        index = 0
        while line:
            item = line.split("\t")
            if len(item) < 4:
                print(line)
                print(index)
            n_id = item[0]
            category = item[1]
            name = item[2]
            # des = item[3].rstrip("\n") if item[3] != "\n" else " "
            # text = category + " " + n_id + " " + name + " " + des
            # text = name + " " + category 
            text = name + " " + category 
            
            texts.append(text)
            id2index[n_id] = index

            index += 1
            line = f.readline()
    
    return id2index, texts




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default="../log_folder")
    parser.add_argument("--log_name", type=str, help="log file name", default="step5.log")
    parser.add_argument('--gpu', type=int, help='gpu device (default: 0)', default=0)
    parser.add_argument("--use_gpu", action="store_true", help="Whether use GPU or not", default=False)
    parser.add_argument('--data_dir', type=str, help='Full path of data folder', default='../data')
    parser.add_argument('--seed', type=int, help='Random seed (default: 1023)', default=1023)
    parser.add_argument("--batch_size", type=int, help="Batch size of bert embedding calculation", default=0)
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default="../data")
    args = parser.parse_args()

    args.batch_size = args.batch_size if args.batch_size else 10

    logger = utils.get_logger(os.path.join(args.log_dir,args.log_name))
    args.logger = logger
    logger.info(args)

    utils.set_random_seed(args.seed)

    print(f"Start Loading data from {args.data_dir}")
    id2index, texts = load_data(args)
    index2id = {value:key for key, value in id2index.items()}

    if not os.path.exists(os.path.join(args.output_folder, "text_embedding")):
        os.makedirs(os.path.join(args.output_folder, "text_embedding"))

    
    # writing files
    with open(os.path.join(args.output_folder, "text_embedding", "ca.json"), "w") as f:
        json.dump(id2index, f)
    with open(os.path.join(args.output_folder, "text_embedding", "index2id.json"), "w") as f:
        json.dump(index2id, f)

    if args.use_gpu and torch.cuda.is_available():
        use_gpu = True
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.set_device(args.gpu)
    elif args.use_gpu:
        print('No GPU is detected in this computer. Use CPU instead.')
        use_gpu = False
        device = 'cpu'
    else:
        use_gpu = False
        device = 'cpu'
    args.use_gpu = use_gpu
    args.device = device

    # tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-base")
    # model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-base")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model.to(args.device)

    ori_embedding = np.zeros([len(texts), 768])

    print(f"Calculating BERT embedding on {args.device} with batch size: {args.batch_size}")

    start_time = time.time()

    for i in range(len(texts) // args.batch_size):
        if (i * batch_size) % 10000 == 0:
            print(f"Finished: {i * args.batch_size} in {time.time() - start_time}")
            start_time = time.time()
        batch_text = texts[i*args.batch_size:(i+1)*args.batch_size]
        batch_embeddings = get_bert_embedding(batch_text, tokenizer, model, args.device)
        ori_embedding[i*args.batch_size:(i+1)*args.batch_size] = batch_embeddings
    
    if (i+1)*args.batch_size < len(texts):
        batch_text = texts[(i+1)*args.batch_size:]
        batch_embeddings = get_bert_embedding(batch_text, tokenizer, model, args.device)
        ori_embedding[(i+1)*args.batch_size:] = batch_embeddings
            
    print("Fitting new embedding with PCA")

    pca = PCA(n_components=100)
    pca_embedding = pca.fit_transform(ori_embedding)

    print("Generating and saving data")

    id2embedding = {}
    for n_id in id2index.keys():
        id2embedding[n_id] = pca_embedding[id2index[n_id]]

    with open(os.path.join(args.output_folder, "text_embedding", "embedding_biobert_namecat.pkl"), 'wb') as f:
        pickle.dump(id2embedding, f, protocol=pickle.HIGHEST_PROTOCOL)
        


    



