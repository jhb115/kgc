# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from typing import Dict
import pickle
import configparser

import torch
from torch import optim
from kbc.datasets import Dataset
from kbc.models import CP, ComplEx, ConvE, Context_CP
from kbc.regularizers import N2, N3, N4
from kbc.optimizers import KBCOptimizer
import os
import numpy as np

# python kbc/learn.py --dataset 'FB15K' --model 'Context_CP' --regularizer 'N3' --max_epoch 1 --max_NB 50 --mkdir True

#For reproducilibility
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

models = ['CP', 'ComplEx', 'ConvE', 'Context_CP']
parser.add_argument(
    '--model', choices=models,
    help="Model in {}".format(models)
)

#Choosing --regularizer 'N0' will disable regularization term
regularizers = ['N0', 'N3', 'N2', 'N4']
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
    help="Factorization rank i.e. Embedding Size"
)

parser.add_argument(
    '--batch_size', default=100, type=int,
    help="Batch Size"
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
    '--learning_rate', default=0.1, type=float,
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

# For Context-based models
parser.add_argument(
    '--max_NB', default=50, type=int,
    help="Number of neighbouring nodes to consider for a give subject node"
)

# Utility related arguments
parser.add_argument(
    '--mkdir', default=0, type=int, choices=[0, 1],
    help='1 if you are running first time (create folders for storing the results)'
)

parser.add_argument(
    '--save_pre_train', default=0, type=int, choices=[0,1],
    help='1 if you wish to pre-train and save the embedding on non-context-based model'
)

parser.add_argument(
    '--load_pre_train', default=0, type=int, choices=[0,1],
    help='1 if you wish to load the saved pre-train the embedding for Context-based model'
)

# Setup parser
args = parser.parse_args()


# Get Dataset
dataset = Dataset(args.dataset)
if args.model in ['CP', 'ComplEx', 'ConvE']:  # For non-context model
    unsorted_examples = torch.from_numpy(dataset.get_train().astype('int64'))
    examples = unsorted_examples
else:  # Get sorted examples for context model
    sorted_data, slice_dic = dataset.get_sorted_train()
    examples = torch.from_numpy(dataset.get_train().astype('int64'))


print(dataset.get_shape())
model = {
    'CP': lambda: CP(dataset.get_shape(), args.rank, args.init),
    'ComplEx': lambda: ComplEx(dataset.get_shape(), args.rank, args.init),
    'ConvE': lambda: ConvE(dataset.get_shape(), args.rank, tuple(args.dropouts), args.use_bias, tuple(args.hw),
                           tuple(args.kernel_size), args.output_channel),
    'Context_CP': lambda: Context_CP(dataset.get_shape(), args.rank, sorted_data, slice_dic,
                                     max_NB=args.max_NB, init_size=args.init, data_name=args.dataset)
}[args.model]()

regularizer = {
    'N0': 'N0',
    'N2': N2(args.reg),
    'N3': N3(args.reg),
    'N4': N4(args.reg)
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

optimizer = KBCOptimizer(model, regularizer, optim_method, args.batch_size)

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


cur_loss = 0
test_i = 0

hits_name = ['_hits@1', '_hits@3', '_hits@10']

train_mrr = []
train_hit1 = []
train_hit3 = []
train_hit10 = []

valid_mrr = []
valid_hit1 = []
valid_hit3 = []
valid_hit10 = []


test_mrr = []
test_hit1 = []
test_hit3 = []
test_hit10 = []

#check if the directory exists
results_folder = '../results/{}/{}'.format(args.model, args.dataset)


# Load pre-trained embeddings
if args.save_pre_train == 1:
    pre_train_folder = '../pre_train/{}/{}'.format('Context_' + args.model, args.dataset)
if args.load_pre_train == 1:
    pre_train_folder = '../pre_train/{}/{}'.format(args.model, args.dataset)
    model.lhs.load_state_dict(torch.load(pre_train_folder + '/lhs.pt'))
    model.rel.load_state_dict(torch.load(pre_train_folder + '/rel.pt'))
    model.rhs.load_state_dict(torch.load(pre_train_folder + '/rhs.pt'))

# make appropriate directories and folders for storing the results
if args.mkdir:
    if not os.path.exists('../results'):
        os.mkdir('../results')
    model_list = ['ComplEx', 'ConvE', 'CP', 'Context_CP', 'Context_ComplEx', 'Context_ConvE']
    dataset_list = ['FB15K', 'FB237', 'WN', 'WN18RR', 'YAGO3-10']

    for each_model in model_list:
        if not os.path.exists('../results/{}'.format(each_model)):
            os.mkdir('../results/{}'.format(each_model))

        for each_data in dataset_list:
            if not os.path.exists('../results/{}/{}'.format(each_model, each_data)):
                os.mkdir('../results/{}/{}'.format(each_model, each_data))

    if not os.path.exists('./debug'):  # for saving debugging files; delete this at the end
        os.mkdir('./debug')

    if args.save_pre_train == 1:
        # this is where the pre-trained emebedding will be saved
        if not os.path.exists('../pre_train'):
            os.mkdir('../pre_train')

        model_list = ['ComplEx', 'ConvE', 'CP']
        dataset_list = ['FB15K', 'FB237', 'WN', 'WN18RR', 'YAGO3-10']

        for each_model in model_list:
            if not os.path.exists('../pre_train/{}'.format('Context_'+each_model)):
                os.mkdir('../pre_train/{}'.format('Context_'+each_model))

            for each_data in dataset_list:
                if not os.path.exists('../pre_train/{}/{}'.format('Context_'+each_model, each_data)):
                    os.mkdir('../pre_train/{}/{}'.format('Context_'+each_model, each_data))

            if not os.path.exists('../pre_train/{}/{}/{}'.format('Context_'+each_model, args.dataset, str(args.rank))):
                os.mkdir('../pre_train/{}/{}/{}'.format('Context_'+each_model, args.dataset, str(args.rank)))

if not os.path.exists(results_folder):
    raise Exception('You do not have folder named:{}'.format(results_folder))

train_no = 1

while os.path.exists(results_folder + '/train' + str(train_no)):
    train_no += 1

folder_name = results_folder + '/train' + str(train_no)
os.mkdir(folder_name)

# Save the configuration file
config = vars(args)

pickle.dump(config, open(folder_name + '/config.p', 'wb'))

config_ini = configparser.ConfigParser()
config_ini['setup'] = {}

for key in config.keys():
    config_ini['setup'][str(key)] = str(config[key])

# Comments about the model
# Flag that we use batch-norm and dropout
config_ini['setup']['only dropout no batch norm'] = 'True'
config_ini['setup']['product score function'] = 'True'

with open(folder_name + '/config.ini', 'w') as configfile:
    config_ini.write(configfile)


for e in range(args.max_epochs):
    print('\n train epoch = ', e+1)
    cur_loss = optimizer.epoch(examples)

    if (e + 1) % args.valid == 0 or (e+1) == args.max_epochs:
        torch.save(model.state_dict(), folder_name + '/model_state.pt')

        if args.save_pre_train:  # save only the embeddings (for pre-training)
            torch.save(model.lhs.state_dict(), pre_train_folder + '/lhs.pt')
            torch.save(model.rel.state_dict(), pre_train_folder + '/rel.pt')
            torch.save(model.rhs.state_dict(), pre_train_folder + '/rhs.pt')
            with open(pre_train_folder + '/config.ini', 'w') as configfile:
                config_ini.write(configfile)

        model.i = 0
        train_results = avg_both(*dataset.eval(model, 'train', 50000))
        model.i = 1
        valid_results = avg_both(*dataset.eval(model, 'valid', -1))

        print("\n\t TRAIN: ", train_results)
        print("\t VALID : ", valid_results)  # change this back

        # Below is Functionality for saving scores but
        # we switch these off for now (during debugging)
        train_mrr.append(train_results['MRR'])

        hits1310 = train_results['hits@[1,3,10]'].numpy()
        train_hit1.append(hits1310[0])
        train_hit3.append(hits1310[1])
        train_hit10.append(hits1310[2])

        valid_mrr.append(valid_results['MRR'])

        hits1310 = valid_results['hits@[1,3,10]'].numpy()
        valid_hit1.append(hits1310[0])
        valid_hit3.append(hits1310[1])
        valid_hit10.append(hits1310[2])

        np.save(folder_name + '/train_mrr', np.array(train_mrr))
        np.save(folder_name + '/train_hit1', np.array(train_hit1))
        np.save(folder_name + '/train_hit3', np.array(train_hit3))
        np.save(folder_name + '/train_hit10', np.array(train_hit10))

        np.save(folder_name + '/valid_mrr', np.array(valid_mrr))
        np.save(folder_name + '/valid_hit1', np.array(valid_hit1))
        np.save(folder_name + '/valid_hit3', np.array(valid_hit3))
        np.save(folder_name + '/valid_hit10', np.array(valid_hit10))

        results = avg_both(*dataset.eval(model, 'test', -1))

        test_mrr.append(results['MRR'])

        hits1310 = results['hits@[1,3,10]'].numpy()

        test_hit1.append(hits1310[0])
        test_hit3.append(hits1310[1])
        test_hit10.append(hits1310[2])

        print("\n\nTEST : ", results)

        np.save(folder_name + '/test_mrr', np.array(test_mrr))
        np.save(folder_name + '/test_hit1', np.array(test_hit1))
        np.save(folder_name + '/test_hit3', np.array(test_hit3))
        np.save(folder_name + '/test_hit10', np.array(test_hit10))

        if args.save_pre_train:
            np.save(pre_train_folder + '/test_mrr', np.array(test_mrr))

        config['e'] = e
        pickle.dump(config, open(folder_name + '/config.p', 'wb'))

        config_ini['setup']['e'] = str(e)

        with open(folder_name + '/config.ini', 'w') as configfile:
            config_ini.write(configfile)

        test_i += 1

        # # For debugging, delete afterwards
        # np.save('./debug/alpha_list', np.array(model.alpha_list))
        # np.save('./debug/e_c_list', np.array(model.e_c_list))
        # np.save('./debug/nb_num', np.array(model.nb_num))
        # np.save('./debug/e_head', np.array(model.e_head))
        if args.model in ['Context_CP', 'Context_ConvE', 'Context_ComplEx']:
            np.save('./debug/forward_g', np.array(model.forward_g))
            np.save('./debug/valid_g', np.array(model.valid_g))
