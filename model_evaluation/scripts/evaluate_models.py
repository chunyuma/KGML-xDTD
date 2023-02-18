import sys, os
import eval_utilities
from eval_mode import DDP_Mode, MOA_Mode
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="The path to data folder", default="../data")
    parser.add_argument("--model_path", type=str, help="The path to model folder", default="../models")
    parser.add_argument("--selected_ddp_models", type=str, nargs="*", default=['kgml_xdtd'], help="What models to evaluate its/their performance for Drug Repurposing Prediction (DDP) mode. Options: 'all','transe','transr','rotate','distmult','complex','analogy','simple','graphsage_link','graphsage_logistic','graphsage_svm','kgml_xdtd_wo_naes','2class_kgml_xdtd','kgml_xdtd'.")
    parser.add_argument("--selected_moa_models", type=str, nargs="*", default=['kgml_xdtd'], help="What models to evaluate its/their performance for Michanism of Action (MOA) mode. Options: 'all', 'multihop', 'kgml_xdtd_wo_dp', 'kgml_xdtd'")
    parser.add_argument("--eval_mode", type=str, default='ddp', help="What mode to do evaluation (Thre choices: 'ddp', 'moa', 'both'). Note that 'ddp' for drug repurposing prediction and 'moa' for mechanism of action path prediction.", choices=['ddp','moa', 'both'])
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size of training data")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU to train model", default=False)
    parser.add_argument('--gpu', type=int, help='gpu device (default: 0) if use_multiple_gpu is False', default=0)
    parser.add_argument("--seed", type=int, help="Manually set initial seed for pytorch", default=1020)
    parser.add_argument('--use_ddp_all', action='store_false', help="In ddp mode, use three different replacement method to calcualte MRR and Hit@k", default=True)
    args = parser.parse_args()

    eval_utilities.set_random_seed(args.seed)
    logger = eval_utilities.get_logger()

    ## check parameters
    if args.use_gpu and torch.cuda.is_available():
        use_gpu = True
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
    elif args.use_gpu:
        logger.info('No GPU is detected in this computer. Use CPU instead.')
        use_gpu = False
        device = 'cpu'
    else:
        use_gpu = False
        device = 'cpu'
    args.use_gpu = use_gpu
    args.device = device


    if not (os.path.exists(args.data_path) and os.path.isdir(args.data_path)):
        logger.error("The given parameter 'data_path' doesn't exist or is not a directory. Please specify the correct path of 'data_path'.")
        exit()
    else:
        logger.info(f"The given parameter 'data_path' is {args.data_path}")

    if not (os.path.exists(args.model_path) and os.path.isdir(args.model_path)):
        logger.error("The given parameter 'model_path' doesn't exist or is not a directory. Please specify the correct path of 'model_path'.")
        exit()
    else:
        logger.info(f"The given parameter 'model_path' is {args.model_path}")

    if args.selected_ddp_models:
        allowable_models = set(["all","transe","transr","rotate","distmult","complex","analogy","simple","graphsage_link","graphsage_logistic","graphsage_svm","kgml_xdtd_wo_naes","2class_kgml_xdtd","kgml_xdtd"])
        args.selected_ddp_models = list(map(lambda x: x.lower(), args.selected_ddp_models))
        temp_diff_models = set(args.selected_ddp_models).difference(allowable_models)
    else:
        logger.error(f"Please use paramter 'selected_ddp_models' to select at least one of models {['all','transe','transr','rotate','distmult','complex','analogy','simple','graphsage_link','graphsage_logistic','graphsage_svm','kgml_xdtd_wo_naes','2class_kgml_xdtd','kgml_xdtd']}.")
        exit()

    if len(temp_diff_models) > 0:
        logger.error(f"Found invalid mdoels: {list(temp_diff_models)}. Only the following ddp models are allowable: 'all','transe','transr','rotate','distmult','complex','analogy','simple','graphsage_link','graphsage_logistic','graphsage_svm','kgml_xdtd_wo_naes','2class_kgml_xdtd','kgml_xdtd'.")
        exit()
    elif 'all' in args.selected_ddp_models:
        logger.info(f"You choose to run 'all' models based on one of your choices in the given parameter 'selected_ddp_models' {args.selected_ddp_models}")
    else:
        logger.info(f"You choose to run the following ddp models: {args.selected_ddp_models}")

    if args.selected_moa_models:
        allowable_models = set(["all","multihop","kgml_xdtd_wo_dp","kgml_xdtd"])
        args.selected_moa_models = list(map(lambda x: x.lower(), args.selected_moa_models))
        temp_diff_models = set(args.selected_moa_models).difference(allowable_models)
    else:
        logger.error(f"Please use paramter 'selected_moa_models' to select at least one of models {['all','multihop','kgml_xdtd_wo_dp','kgml_xdtd']}.")
        exit()

    if len(temp_diff_models) > 0:
        logger.error(f"Found invalid mdoels: {list(temp_diff_models)}. Only the following moa models are allowable: 'all','multihop','kgml_xdtd_wo_dp','kgml_xdtd'.")
        exit()
    elif 'all' in args.selected_moa_models:
        logger.info(f"You choose to run 'all' models based on one of your choices in the given parameter 'selected_moa_models' {args.selected_moa_models}")
    else:
        logger.info(f"You choose to run the following moa models: {args.selected_moa_models}")

    logger.info(f"You choose to evaluate the models using '{args.eval_mode}' mode.")
    if args.eval_mode == 'ddp' and not args.use_ddp_all:
        logger.info("In ddp mode, use three different replacement method to calcualte MRR and Hit@k")

    if args.eval_mode == 'both':
        ## running both evaluation modes
        ddp_evaluator = DDP_Mode(args, logger)
        ddp_evaluator.do_evaluation(args.selected_ddp_models, args.use_ddp_all)
        moa_evaluator = MOA_Mode(args, logger)
        moa_evaluator.do_evaluation(args.selected_moa_models)

    elif args.eval_mode == 'ddp':
        ## running 'ddp' evaluation mode
        ddp_evaluator = DDP_Mode(args, logger)
        ddp_evaluator.do_evaluation(args.selected_ddp_models, args.use_ddp_all)
    else:
        ## running 'moa' evaluation mode
        moa_evaluator = MOA_Mode(args, logger)
        moa_evaluator.do_evaluation(args.selected_moa_models)
