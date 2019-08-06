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
from kbc.models import CP, ComplEx, Context_CP, Context_ComplEx
from kbc.regularizers import N2, N3, N4
from kbc.optimizers import KBCOptimizer
import os
import numpy as np

# For reproducilibility
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

'''
List of Hyperparameters to tune:
rank = [100, 200, 400]
batch_size = [300]
learning_rate = [0.05]
g_weight = [0, 0.03]
max_NB = [10, 100]
'''

big_datasets = ['FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10']
datasets = big_datasets

parser = argparse.ArgumentParser(
    description="Relational learning contraption"
)

parser.add_argument(
    '--g_weight', type=float, default=0.,
    help='weights on the g regularization term'
)

parser.add_argument(
    '--dataset', choices=datasets,
    help="Dataset in {}".format(datasets)
)

models = ['CP', 'ComplEx', 'Context_CP', 'Context_ComplEx']
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
    '--save_pre_train', default=0, type=int, choices=[0, 1],
    help='1 if you wish to pre-train and save the embedding on non-context-based model'
)

parser.add_argument(
    '--load_pre_train', default=0, type=int, choices=[0, 1],
    help='1 if you wish to load the saved pre-train the embedding for Context-based model'
)

parser.add_argument(
    '--ascending', default=1, type=int, choices=[-1, 1],
    help='1 if you wish to consider neighborhood degrees in ascending order, -1 otherwise'
)

# Setup parser
args = parser.parse_args()

# Get Dataset
dataset = Dataset(args.dataset)
if args.model in ['CP', 'ComplEx']:
    unsorted_examples = torch.from_numpy(dataset.get_train().astype('int64'))
    examples = unsorted_examples
else:
    sorted_data, slice_dic = dataset.get_sorted_train()
    examples = torch.from_numpy(dataset.get_train().astype('int64'))

model = {
    'CP': lambda: CP(dataset.get_shape(), args.rank, args.init),
    'ComplEx': lambda: ComplEx(dataset.get_shape(), args.rank, args.init),
    'Context_CP': lambda: Context_CP(dataset.get_shape(), args.rank, sorted_data, slice_dic,
                                     max_NB=args.max_NB, init_size=args.init, data_name=args.dataset,
                                     ascending=args.ascending),
    'Context_ComplEx': lambda: Context_ComplEx(dataset.get_shape(), args.rank, sorted_data, slice_dic,
                                               max_NB=args.max_NB, init_size=args.init, data_name=args.dataset,
                                               ascending=args.ascending)
}[args.model]()

regularizer = {
    'N0': 'N0',
    'N2': N2(args.reg),
    'N3': N3(args.reg),
    'N4': N4(args.reg, g_weight=args.g_weight)
}[args.regularizer]

device = 'cuda'
model.to(device)

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

hits_name = ['_hits@1', '_hits@3', '_hits@10']

train_mrr = []
train_hit1 = []
train_hit3 = []
train_hit10 = []

test_mrr = []
test_hit1 = []
test_hit3 = []
test_hit10 = []

results_folder = '../results/{}/{}'.format(args.model, args.dataset)

# Load pre-trained embeddings
if args.save_pre_train == 1:
    pre_train_folder = '../pre_train/{}/{}/{}'.format('Context_' + args.model, args.dataset, str(args.rank))

if args.load_pre_train == 1:
    pre_train_folder = '../pre_train/{}/{}/{}'.format(args.model, args.dataset, str(args.rank))
    if args.model == 'Context_CP':
        model.lhs.load_state_dict(torch.load(pre_train_folder + '/lhs.pt'))
        model.rel.load_state_dict(torch.load(pre_train_folder + '/rel.pt'))
        model.rhs.load_state_dict(torch.load(pre_train_folder + '/rhs.pt'))
    elif args.model == 'Context_ComplEx':
        # model.embeddings = torch.load(pre_train_folder + '/embeddings.pt')
        model.embeddings[0].load_state_dict(torch.load(pre_train_folder + '/entity.pt'))
        model.embeddings[1].load_state_dict(torch.load(pre_train_folder + '/relation.pt'))

if args.mkdir:
    if not os.path.exists('../results'):
        os.mkdir('../results')
    model_list = ['ComplEx', 'CP', 'Context_CP', 'Context_ComplEx']
    dataset_list = ['FB15K', 'FB237', 'WN', 'WN18RR', 'YAGO3-10']

    for each_model in model_list:
        if not os.path.exists('../results/{}'.format(each_model)):
            os.mkdir('../results/{}'.format(each_model))

        for each_data in dataset_list:
            folder_name = '../results/{}/{}'.format(each_model, each_data)
            if not os.path.exists(folder_name):
                os.mkdir('../results/{}/{}'.format(each_model, each_data))
            # check if the summary configuration file exists or not,
            if not os.path.exists(folder_name + '/summary_config.ini'):
                # make config summary file
                summary_config = configparser.ConfigParser()
                summary_config['summary'] = {'model': each_model,
                                             'dataset': each_data,
                                             'best_mrr': '0',
                                             'best_hits@10': '0'}

                with open(folder_name + '/summary_config.ini', 'w') as configfile:
                    summary_config.write(configfile)

    if args.save_pre_train == 1:
        # this is where the pre-trained emebedding will be saved
        if not os.path.exists('../pre_train'):
            os.mkdir('../pre_train')

        model_list = ['ComplEx', 'CP']
        dataset_list = ['FB15K', 'FB237', 'WN', 'WN18RR', 'YAGO3-10']

        for each_model in model_list:
            folder_name = '../pre_train/{}'.format('Context_'+each_model)
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)

            for each_data in dataset_list:
                if not os.path.exists('{}/{}'.format(folder_name, each_data)):
                    os.mkdir('{}/{}'.format(folder_name, each_data))

                if not os.path.exists(folder_name + '/summary_config.ini'):
                    summary_config = configparser.ConfigParser()
                    summary_config['summary'] = {'model': each_model, 'dataset': each_data}
                    with open(folder_name + '/summary_config.ini', 'w') as configfile:
                        summary_config.write(configfile)

        if not os.path.exists('{}/{}/{}'.format(folder_name, each_data, str(args.rank))):
            os.mkdir('{}/{}/{}'.format(folder_name, each_data, str(args.rank)))

# We load our summary configuration file:
summary_config = configparser.ConfigParser()
summary_config.read('{}/summary_config.ini'.format(results_folder))

train_no = 1

while os.path.exists(results_folder + '/train' + str(train_no)):
    train_no += 1

train_no = 'train' + str(train_no)
folder_name = '{}/{}'.format(results_folder, train_no)
os.mkdir(folder_name)

summary_config[train_no] = {}

config = vars(args)
pickle.dump(config, open(folder_name + '/config.p', 'wb'))

for key in config.keys():
    summary_config[train_no][str(key)] = str(config[key])

summary_config['summary']['Currently_running_experiment'] = '{} on {}'.format(args.model, args.dataset)

with open(folder_name + '/summary_config.ini', 'w') as configfile:
    summary_config.write(configfile)



forward_g = []
test_g = []
forward_alpha = []
test_alpha = []
best_model_flag = 0

for e in range(args.max_epochs):
    print('\n train epoch = ', e+1)
    cur_loss = optimizer.epoch(examples)

    if (e + 1) % args.valid == 0 or (e+1) == args.max_epochs:

        if args.save_pre_train:  # save only the embeddings (for pre-training)
            if args.model == 'CP':
                torch.save(model.lhs.state_dict(), pre_train_folder + '/lhs.pt')
                torch.save(model.rel.state_dict(), pre_train_folder + '/rel.pt')
                torch.save(model.rhs.state_dict(), pre_train_folder + '/rhs.pt')
                with open(pre_train_folder + '/summary_config', 'w') as configfile:
                    summary_config.write(configfile)
            elif args.model == 'ComplEx':
                torch.save(model.embeddings[0].state_dict(), pre_train_folder+'/entity.pt')
                torch.save(model.embeddings[1].state_dict(), pre_train_folder+'/relation.pt')

        train_results = avg_both(*dataset.eval(model, 'train', 50000))

        if best_model_flag == 1 and ((e+1) % (args.valid*3) == 0):
            forward_g.append(model.g.clone().data.cpu().numpy())
            forward_alpha.append(model.alpha.clone().data.cpu().numpy())


        print("\n\t TRAIN: ", train_results)

        train_mrr.append(train_results['MRR'])

        hits1310 = train_results['hits@[1,3,10]'].numpy()
        train_hit1.append(hits1310[0])
        train_hit3.append(hits1310[1])
        train_hit10.append(hits1310[2])

        summary_config[train_no]['curr_train_hit10'] = str(hits1310[2])
        summary_config[train_no]['curr_train_mrr'] = str(train_results['MRR'])

        np.save(folder_name + '/train_mrr', np.array(train_mrr))
        np.save(folder_name + '/train_hit1', np.array(train_hit1))
        np.save(folder_name + '/train_hit3', np.array(train_hit3))
        np.save(folder_name + '/train_hit10', np.array(train_hit10))

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

        max_test_mrr = max(np.array(test_mrr))
        max_test_hits = max(np.array(test_hit10))
        summary_config[train_no]['max_test_hit10'] = str(max_test_hits)
        summary_config[train_no]['max_test_mrr'] = str(max_test_mrr)

        if max_test_mrr >= float(summary_config['summary']['best_mrr']):
            best_model_flag = 1
            summary_config['summary']['best_train_no'] = str(train_no)
            summary_config['summary']['best_mrr'] = str(max_test_mrr)
            summary_config['summary']['best_hits@10'] = str(max_test_hits)
            torch.save(model.state_dict(), folder_name + '/model_state.pt')

        if best_model_flag == 1 and ((e+1) % (args.valid*3) == 0):
            test_g.append(model.g.clone().data.cpu().numpy())
            test_alpha.append(model.alpha.clone().data.cpu().numpy())

            # save the model if the mrr or hits@10 is best among all

        config['e'] = e
        pickle.dump(config, open(folder_name + '/config.p', 'wb'))

        summary_config[train_no]['e'] = str(e)

        with open(folder_name + '/summary_config.ini', 'w') as configfile:
            summary_config.write(configfile)

eps = 0.002
if (max(np.array(test_mrr)) == np.array(test_mrr)[-1]) and (abs(np.array(test_mrr)[-1] - np.array(test_mrr)[-2]) > eps):
    torch.save(model.state_dict(), folder_name + '/model_state.pt')
    summary_config['summary']['{} not fully trained'.format(train_no)] = 'True'

    with open(folder_name + '/summary_config.ini', 'w') as configfile:
        summary_config.write(configfile)
