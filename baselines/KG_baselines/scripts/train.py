import os
import argparse
import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE, TransR, TransH, ComplEx, DistMult, RotatE, SimplE, Analogy
from openke.module.loss import MarginLoss, SoftplusLoss, SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    default='TransE',
    type=str,
    help="The model to use",
)
parser.add_argument(
    "--data",
    default='./results/processed/',
    type=str,
    help="The dir where train/test/val files locates",
)
parser.add_argument(
    "--dim",
    default=100,
    type=int,
    help="Dimension of the model",
)
parser.add_argument(
    "--batch_size",
    default=10000,
    type=int,
    help="Batch size in training",
)
parser.add_argument(
    "--epoch",
    default=1000,
    type=int,
    help="Number of epoch for training",
)
parser.add_argument(
    "--checkpoint",
    default='./results/checkpoints/',
    type=str,
    help="Output Checkpoint Folder",
)

args = parser.parse_args()

model2hyperparam = {
    'transh': [0.5, 'sgd'],
    'transr': [1.0, 'sgd'],
    'transe': [1.0, 'sgd'],
    'complex': [0.5, 'adagrad'],
    'distmult': [0.5, 'adagrad'],
    'simple': [0.5, 'adagrad'],
    'analogy': [0.5, 'adagrad'],
    'rotate': [2e-5, 'adam'],
}

if not os.path.exists(args.data):
    os.makedirs(args.data)

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)

lr, optimizer = model2hyperparam[args.model]
MODEL_NAME = os.path.join(args.checkpoint, '{}_lr{}_{}_bs{}_ep{}.ckpt'.format(args.model, lr, optimizer, args.batch_size, args.epoch))

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = args.data, 
	batch_size = args.batch_size,
	threads = 24, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader(args.data, "link")

if args.model == 'transe':
    # define the model
    kg_model = TransE(
        ent_tot = train_dataloader.get_ent_tot(),
        rel_tot = train_dataloader.get_rel_tot(),
        dim = args.dim, 
        p_norm = 1, 
        norm_flag = True)


    model = NegativeSampling(
        model = kg_model, 
        loss = MarginLoss(margin = 5.0),
        batch_size = train_dataloader.get_batch_size())

elif args.model == 'transr':
    kg_model = TransR(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim_e = args.dim,
	dim_r = args.dim,
	p_norm = 1, 
	norm_flag = True,
	rand_init = True)

    model = NegativeSampling(
        model = kg_model,
        loss = MarginLoss(margin = 4.0),
        batch_size = train_dataloader.get_batch_size()
    )

elif args.model == 'simple':
    kg_model = SimplE(
        ent_tot = train_dataloader.get_ent_tot(),
        rel_tot = train_dataloader.get_rel_tot(),
        dim = args.dim
    )

    model = NegativeSampling(
        model = kg_model, 
        loss = SoftplusLoss(),
        batch_size = train_dataloader.get_batch_size(), 
        regul_rate = 1.0
    )

elif args.model == 'analogy':
    kg_model = Analogy(
        ent_tot = train_dataloader.get_ent_tot(),
        rel_tot = train_dataloader.get_rel_tot(),
        dim = args.dim
    )

    model = NegativeSampling(
        model = kg_model, 
        loss = SoftplusLoss(),
        batch_size = train_dataloader.get_batch_size(), 
        regul_rate = 1.0
    )

elif args.model == 'transh':
    kg_model = TransH(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = args.dim, 
	p_norm = 1, 
	norm_flag = True)

    model = NegativeSampling(
        model = kg_model, 
        loss = MarginLoss(margin = 4.0),
        batch_size = train_dataloader.get_batch_size()
    )

elif args.model == 'complex':
    kg_model = ComplEx(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = args.dim)

    model = NegativeSampling(
        model = kg_model, 
        loss = SoftplusLoss(),
        batch_size = train_dataloader.get_batch_size(), 
        regul_rate = 1.0)


elif args.model == 'distmult':
    kg_model = DistMult(
        ent_tot = train_dataloader.get_ent_tot(),
        rel_tot = train_dataloader.get_rel_tot(),
        dim = args.dim
    )

    # define the loss function
    model = NegativeSampling(
        model = kg_model, 
        loss = SoftplusLoss(),
        batch_size = train_dataloader.get_batch_size(), 
        regul_rate = 1.0
    )

elif args.model == 'rotate':
    kg_model = RotatE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = args.dim,
	margin = 6.0,
	epsilon = 2.0,
)

# define the loss function
    model = NegativeSampling(
	model = kg_model, 
	loss = SigmoidLoss(adv_temperature = 2),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 0.0
)

else:
    raise ValueError("Choose args.model from ['transe', 'complex', 'distmult']")

trainer = Trainer(model = model, data_loader = train_dataloader, train_times = args.epoch, alpha = lr, use_gpu = True, opt_method = optimizer)
trainer.run()
kg_model.save_checkpoint(MODEL_NAME)
