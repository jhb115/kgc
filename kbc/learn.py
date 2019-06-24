# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from typing import Dict

import torch
from torch import optim

from kbc.datasets import Dataset
from kbc.models import CP, ComplEx, ConvE
from kbc.regularizers import N2, N3
from kbc.optimizers import KBCOptimizer
import os
import pickle
import pandas as pd

import numpy as np

#Fix the random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

big_datasets = ['FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10']
datasets = big_datasets

parser = argparse.ArgumentParser(
    description="Relational learning contraption"
)

parser.add_argument(
    '--dataset', choices=datasets,
    help="Dataset in {}".format(datasets)
)

models = ['CP', 'ComplEx', 'ConvE']
parser.add_argument(
    '--model', choices=models,
    help="Model in {}".format(models)
)

regularizers = ['N0', 'N3', 'N2']
parser.add_argument(
    '--regularizer', choices=regularizers, default='N3',
    help="Regularizer in {}".format(regularizers)
)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument(
    '--optimizer', choices=optimizers, default='Adagrad',
    help="Optimizer in {}".format(optimizers)
)

parser.add_argument(
    '--max_epochs', default=50, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid', default=3, type=float,
    help="Number of epochs before valid."
)
parser.add_argument(
    '--rank', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=100, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--reg', default=0, type=float,
    help="Regularization weight"
)
parser.add_argument(
    '--init', default=1e-1, type=float,
    help="Initial scale"
)
parser.add_argument(
    '--learning_rate', default=10, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--decay1', default=0.9, type=float,
    help="decay rate for the first moment estimate in Adam"
)
parser.add_argument(
    '--decay2', default=0.999, type=float,
    help="decay rate for second moment estimate in Adam"
)

# Parser argument for ConvE
# Dropout
parser.add_argument(
    '--dropouts', default=(0.3, 0.3, 0.3), type=float,
    help="Dropout rates for each layer in ConvE"
)
# Boolean for the bias in ConvE layers
parser.add_argument(
    '--use_bias', default=True, type=bool,
    help="Using or not using bias for the ConvE layers"
)

parser.add_argument(
    '--kernel_size', default=(3, 3), nargs='+', type=int,
    help="Kernel Size"
)

parser.add_argument(
    '--output_channel', default=32, type=int,
    help="Number of output channel"
)

parser.add_argument(
    '--hw', default=(10, 20), nargs='+', type=int,
    help="False or (Height, Width) shape for 2D reshaping entity embedding"
)

loss_choices = ['Multi', 'Binary']
parser.add_argument(
    '--loss', default='Multi', type=str, choices=loss_choices,
    help="Choose Binary or Multi for cross entropy loss"
)


args = parser.parse_args()

#Example Run:
#!python kbc/learn.py --dataset 'FB15K' --model 'ConvE' --rank 200 --max_epochs 3 --hw 0 0 --kernel_size 3 3 --output_channel 32
#!python kbc/learn.py --dataset 'FB15K' --model 'ConvE' --rank 200 --max_epochs 3 --hw 0 0 --kernel_size 3 3 --output_channel 32 --regularizer 'N0'

#Choosing --regularizer 'N0' will disable regularization term

if args.model == 'ConvE':
    hw = tuple(args.hw)
    kernel_size = tuple(args.kernel_size)
    dropouts = tuple(args.dropouts)

dataset = Dataset(args.dataset)
examples = torch.from_numpy(dataset.get_train().astype('int64'))

print(dataset.get_shape())
model = {
    'CP': lambda: CP(dataset.get_shape(), args.rank, args.init),
    'ComplEx': lambda: ComplEx(dataset.get_shape(), args.rank, args.init),
    'ConvE': lambda: ConvE(dataset.get_shape(), args.rank, dropouts, args.use_bias, hw, kernel_size,
                           args.output_channel)
}[args.model]()

regularizer = {
    'N0': 'N0',
    'N2': N2(args.reg),
    'N3': N3(args.reg),
}[args.regularizer]

device = 'cuda'
model.to(device)

if args.model == "ConvE":
    model.init()

optim_method = {
    'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
    'Adam': lambda: optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.decay1, args.decay2)),
    'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate)
}[args.optimizer]()

optimizer = KBCOptimizer(model, regularizer, optim_method, args.batch_size, loss_type=args.loss)

def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MRR': m, 'hits@[1,3,10]': h}

# e.g. run
# python kbc/learn.py --dataset 'WN18RR' --model 'ConvE'
# python kbc/learn.py --dataset 'WN18RR' --model 'ConvE' --rank 200 --learning_rate 0.1 --max_epochs 3 --loss 'Binary' --regularizer 'N0'
# python kbc/learn.py --dataset 'WN18RR' --model 'ConvE' --rank 200 --learning_rate 0.1 --max_epochs 3 --loss 'Multi' --regularizer 'N3'
# python kbc/learn.py --dataset 'WN18RR' --model 'ComplEx' --rank 200 -- learning_rate 0.1 --max_epochs 3

# include information of
# model_name, rank, learning_rate, regularization, reg,

cur_loss = 0
for e in range(args.max_epochs):
    cur_loss = optimizer.epoch(examples)

    if (e + 1) % args.valid == 0:
        valid, test, train = [
            avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
            for split in ['valid', 'test', 'train']
        ]


        print("\t TRAIN: ", train)
        print("\t VALID : ", valid)

    if (e+1) % 10 == 0:
        results = dataset.eval(model, 'test', -1)
        results = avg_both(results)


        print("\n\nTEST : ", results)

    if (e+1)% 20 == 0:
        # save the evaluated scores





print("\n\nTEST : ", results)
