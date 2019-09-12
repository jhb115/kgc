import argparse
from typing import Dict
import pickle
import configparser

import torch
from torch import optim
from kbc.datasets import Dataset
from kbc.models import CP, ComplEx, ContExt
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

datasets = ['WN18RR', 'FB237', 'YAGO3-10']

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

models = ['CP', 'ComplEx', 'ContExt']
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

parser.add_argument(
    '--dropout_1', default=0.5, type=float,
    help='Dropout on the first linear projection layer for query vector, used in v3'
)

parser.add_argument(
    '--dropout_g', default=0.3, type=float,
    help='Dropout on the g, used in v3'
)

parser.add_argument(
    '--n_freeze', default=0, type=int,
    help='Number of training epochs you wish to freeze the original embedding'
)

parser.add_argument(
    '--evaluation_mode', default=0, type=int, choices=[0, 1],
    help='Whther to get an attention mask or not'
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
    'ContExt': lambda: ContExt(dataset.get_shape(), args.rank, sorted_data, slice_dic,
                               max_NB=args.max_NB, init_size=args.init, data_name=args.dataset,
                               ascending=args.ascending, dropout_1=args.dropout_1,
                               dropout_g=args.dropout_g, evaluation_mode=args.evaluation_mode),
}[args.model]()

regularizer = {
    'N0': 'N0',
    'N2': N2(args.reg),
    'N3': N3(args.reg),
    'N4': N4(args.reg, g_weight=args.g_weight)
}[args.regularizer]

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')
model.to(device)

# Freeze the embeddings for n_freeze epochs
if args.n_freeze > 0:
    if args.model in ['Context_CP', 'Context_CP_v2']:
        model.lhs.weight.requires_grad = False
        model.rel.weight.requires_grad = False
        model.rh.weight.requires_grad = False

    elif args.model in ['ContExt']:
        for i in range(2):
            model.embeddings[i].weight.requires_grad = False

optim_method = {
    'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
    'RMSprop': lambda: optim.RMSProp(model.parameters(), lr=args.learning_rate),
    'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate)
}[args.optimizer]()

# print('Model state:')
# for param_tensor in model.state_dict():
#     print(f'\t{param_tensor}\t{model.state_dict()[param_tensor].size()}')

optimizer = KBCOptimizer(model, regularizer, optim_method, args.batch_size, n_freeze=args.n_freeze)

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

config_folder = '../results/{}/{}'.format(args.model, args.dataset)

if not os.path.exists('../results'):
    os.mkdir('../results')

model_list = ['ComplEx', 'CP', 'ContExt']
dataset_list = ['FB237', 'WN18RR', 'YAGO3-10']

# For actual model
# saves file summary_config.ini to ../results/model/data
for each_model in model_list:
    if not os.path.exists('../results/{}'.format(each_model)):
        os.mkdir('../results/{}'.format(each_model))

    for each_data in dataset_list:
        folder_name = '../results/{}/{}'.format(each_model, each_data)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        # check if the summary configuration file exists or not,
        if not os.path.exists(folder_name + '/summary_config.ini'):
            # make config summary file
            summary_config = configparser.ConfigParser()
            summary_config['summary'] = {'model': each_model,
                                         'dataset': each_data,
                                         'best_mrr': '0',
                                         'best_hits@10': '0',
                                         'best_train_no': '?'}

            with open(folder_name + '/summary_config.ini', 'w') as configfile:
                summary_config.write(configfile)

# Configuration for pre-training
if not os.path.exists('../pre_train'):
    os.mkdir('../pre_train')

model_list = ['ComplEx', 'CP']
dataset_list = ['FB237', 'WN18RR', 'YAGO3-10']

for each_model in model_list:
    folder_name = '../pre_train/{}'.format('Context_'+each_model)
    # folder_name = '../pre_train/model
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    for each_data in dataset_list:
        if not os.path.exists('{}/{}'.format(folder_name, each_data)):
            os.mkdir('{}/{}'.format(folder_name, each_data))
            # ../pre_train/model/data

        # create ../pre_train/model/data/summary_config.ini
        if not os.path.exists('{}/{}/{}'.format(folder_name, each_data, 'summary_config.ini')):
            summary_config = configparser.ConfigParser()
            summary_config['summary'] = {'model': each_model, 'dataset': each_data}
            with open('{}/{}/{}'.format(folder_name, each_data, 'summary_config.ini'), 'w') as configfile:
                summary_config.write(configfile)

run_pre_train_flag = 0


# Need to consider the case when args.model == 'Context_CP
pre_model_name = {'Contact': 'Contact',
                  'ContExt': 'ContExt'}[args.model]

if args.load_pre_train == 1:
    pre_train_folder = '../pre_train/{}/{}/{}'.format(pre_model_name, args.dataset, str(args.rank))
    # ../pre_train/model/data/rank

    if not os.path.exists(pre_train_folder):
        os.mkdir(pre_train_folder)
        run_pre_train_flag = 1

    if args.model == 'Contact':
        if os.path.exists(pre_train_folder + '/lhs.pt'):
            model.lhs.load_state_dict(torch.load(pre_train_folder + '/lhs.pt'))
            model.rel.load_state_dict(torch.load(pre_train_folder + '/rel.pt'))
            model.rhs.load_state_dict(torch.load(pre_train_folder + '/rhs.pt'))
        else:
            run_pre_train_flag = 1
            pre_train_args = {'model': 'CP', 'regularizer': 'N3', 'max_epoch': 80, 'batch_size': 300,
                              'save_pre_train': 1, 'learning_rate': 0.1, 'reg': 0.1, 'dataset': args.dataset,
                              'rank': args.rank, 'init': args.init}

            pre_train_dataset = Dataset(args.dataset)
            unsorted_examples = torch.from_numpy(pre_train_dataset.get_train().astype('int64'))
            pre_train_model = CP(pre_train_dataset.get_shape(), args.rank, args.init)
            pre_train_regularizer = N3(pre_train_args['reg'])
            device = 'cuda'
            pre_train_model.to(device)
            pre_train_optim = optim.Adagrad(pre_train_model.parameters(), lr=pre_train_args['learning_rate'])
            pre_train_optimizer = KBCOptimizer(pre_train_model, pre_train_regularizer, pre_train_optim,
                                               pre_train_args['batch_size'])

    elif args.model == 'ContExt':
        if os.path.exists(pre_train_folder + '/entity.pt'):
            # model.embeddings = torch.load(pre_train_folder + '/embeddings.pt')
            model.embeddings[0].load_state_dict(torch.load(pre_train_folder + '/entity.pt'))
            model.embeddings[1].load_state_dict(torch.load(pre_train_folder + '/relation.pt'))
        else:
            run_pre_train_flag = 1
            # Does not have a pre-trained embedding. Need to run pre-training ComplEx

            pre_train_args = {'model': 'ComplEx', 'regularizer': 'N3', 'max_epoch': 80, 'save_pre_train': 1}

            if args.dataset == 'FB237' or args.dataset == 'FB15K':
                pre_train_args['learning_rate'] = 0.1
                pre_train_args['batch_size'] = 100
                pre_train_args['reg'] = 0.05
            elif args.dataset == 'WN18RR' or args.dataset == 'WN18RR':
                pre_train_args['learning_rate'] = 0.1
                pre_train_args['batch_size'] = 100
                pre_train_args['reg'] = 0.1
            elif args.dataset == 'YAGO3-10':
                pre_train_args['learning_rate'] = 0.1
                pre_train_args['batch_size'] = 1000
                pre_train_args['reg'] = 0.005

            pre_train_args['rank'] = args.rank
            pre_train_args['init'] = args.init

            pre_train_dataset = Dataset(args.dataset)
            unsorted_examples = torch.from_numpy(pre_train_dataset.get_train().astype('int64'))
            pre_train_model = ComplEx(pre_train_dataset.get_shape(), args.rank, args.init)
            pre_train_regularizer = N3(pre_train_args['reg'])
            device = 'cuda'
            pre_train_model.to(device)
            pre_train_optim = optim.Adagrad(pre_train_model.parameters(), lr=pre_train_args['learning_rate'])
            pre_train_optimizer = KBCOptimizer(pre_train_model, pre_train_regularizer, pre_train_optim,
                                               pre_train_args['batch_size'])

# summary_config is config file for actual model
summary_config = configparser.ConfigParser()
# load ../results/model/dataset/summary_config.ini
summary_config.read('../results/{}/{}/summary_config.ini'.format(args.model, args.dataset))

train_no = 1

while os.path.exists('../results/{}/{}/train{}'.format(args.model, args.dataset, str(train_no))):
    train_no += 1

train_no = 'train' + str(train_no)
os.mkdir('../results/{}/{}/{}'.format(args.model, args.dataset, train_no))
config = vars(args)
pickle.dump(config, open('../results/{}/{}/{}/config.p'.format(args.model, args.dataset, train_no), 'wb'))

summary_config[train_no] = {}
for key in config.keys():
    summary_config[train_no][str(key)] = str(config[key])

summary_config['summary']['Currently_running_experiment'] = '{} on {}'.format(args.model, args.dataset)

with open('../results/{}/{}/summary_config.ini'.format(args.model, args.dataset), 'w') as configfile:
    summary_config.write(configfile)


# For running pre-training
if run_pre_train_flag:
    pre_train_config = configparser.ConfigParser()
    pre_train_config_folder = '../pre_train/{}/{}'.format(pre_model_name, args.dataset)
    pre_train_config.read(pre_train_config_folder + '/summary_config.ini')

    # For each dataset and for each model
    pre_train_config[train_no] = {}

    for key in pre_train_args:
        pre_train_config[train_no][str(key)] = str(pre_train_args[key])

    with open(pre_train_config_folder + '/summary_config.ini', 'w') as configfile:
        pre_train_config.write(configfile)

    pre_train_folder = '../pre_train/{}/{}/{}'.format(pre_model_name, args.dataset, str(args.rank))

    train_mrr = []
    train_hit10 = []
    test_mrr = []
    test_hit10 = []

    pre_train_config[train_no]['best_mrr'] = str(0)

    for e in range(pre_train_args['max_epoch']):
        print('pre_train epoch = ', e)
        cur_loss = pre_train_optimizer.epoch(examples)

        if (e + 1) % args.valid == 0 or (e + 1) == args.max_epochs:
            train_results = avg_both(*pre_train_dataset.eval(pre_train_model, 'train', 50000))
            train_mrr.append(train_results['MRR'])
            hits1310 = train_results['hits@[1,3,10]'].numpy()
            train_hit10.append(hits1310[2])

            results = avg_both(*pre_train_dataset.eval(pre_train_model, 'test', -1))
            test_mrr.append(results['MRR'])
            hits1310 = results['hits@[1,3,10]'].numpy()
            test_hit10.append(hits1310[2])

            pre_train_save_folder = '../pre_train/{}/{}/{}'.format(pre_model_name, args.dataset, str(args.rank))

            np.save('{}/{}'.format(pre_train_folder, 'train_mrr'), np.array(train_mrr))
            np.save('{}/{}'.format(pre_train_folder, 'train_hit10'), np.array(train_hit10))
            np.save('{}/{}'.format(pre_train_folder, 'test_mrr'), np.array(test_mrr))
            np.save('{}/{}'.format(pre_train_folder, 'test_hit10'), np.array(test_hit10))

        pre_train_config[train_no]['e'] = str(e)

        with open(pre_train_config_folder + '/summary_config.ini', 'w') as configfile:
            pre_train_config.write(configfile)

    max_test_mrr = max(np.array(test_mrr))
    max_test_hit10 = np.array(test_hit10)[np.argmax(np.array(test_mrr))]

    pre_train_config[train_no]['best_mrr'] = str(max_test_mrr)
    pre_train_config[train_no]['best_test10'] = str(max_test_hit10)

    with open(pre_train_config_folder + '/summary_config.ini', 'w') as configfile:
        pre_train_config.write(configfile)

    if pre_train_args['model'] == 'CP':
        torch.save(pre_train_model.lhs.state_dict(), pre_train_folder + '/lhs.pt')
        torch.save(pre_train_model.rel.state_dict(), pre_train_folder + '/rel.pt')
        torch.save(model.rhs.state_dict(), pre_train_folder + '/rhs.pt')
    elif pre_train_args['model'] == 'ComplEx':
        torch.save(pre_train_model.embeddings[0].state_dict(), pre_train_folder + '/entity.pt')
        torch.save(pre_train_model.embeddings[1].state_dict(), pre_train_folder + '/relation.pt')

    del pre_train_dataset
    del unsorted_examples
    del pre_train_model
    del pre_train_optim
    del pre_train_optimizer

    if args.model == 'Contact':
        if os.path.exists(pre_train_folder + 'lhs.pt'):
            model.lhs.load_state_dict(torch.load(pre_train_folder + '/lhs.pt'))
            model.rel.load_state_dict(torch.load(pre_train_folder + '/rel.pt'))
            model.rhs.load_state_dict(torch.load(pre_train_folder + '/rhs.pt'))

    elif args.model == 'ContExt':
        if os.path.exists(pre_train_folder + 'entity.pt'):
            model.embeddings[0].load_state_dict(torch.load(pre_train_folder + '/entity.pt'))
            model.embeddings[1].load_state_dict(torch.load(pre_train_folder + '/relation.pt'))


# Relevant variables to store
forward_g = []
test_g = []
forward_alpha = []
test_alpha = []
forward_target_o = []
test_target_o = []

forward_nb_index = []
forward_spo_index = []
test_nb_index = []
test_spo_index = []

best_model_flag = 0

hits_name = ['_hits@1', '_hits@3', '_hits@10']

train_mrr = []
train_hit1 = []
train_hit3 = []
train_hit10 = []

test_mrr = []
test_hit1 = []
test_hit3 = []
test_hit10 = []

folder_name = '../results/{}/{}/{}'.format(args.model, args.dataset, train_no)

for e in range(args.max_epochs):

    print('\n train epoch = ', e+1)

    if e == args.n_freeze and args.n_freeze > 0:
        optimizer.freeze_flag = 0

    cur_loss = optimizer.epoch(examples)

    if (e + 1) % args.valid == 0 or (e+1) == args.max_epochs:

        train_results = avg_both(*dataset.eval(model, 'train', 50000))

        forward_g.append(model.g.clone().data.cpu().numpy())
        forward_alpha.append(model.alpha.clone().data.cpu().numpy())

        if (e+1) % (args.valid * 3) == 0:
            np.save(folder_name + '/forward_g', np.array(forward_g))
            np.save(folder_name + '/forward_alpha', np.array(forward_alpha))

            if args.evaluation_mode:
                forward_nb_index.append(model.index_array)
                forward_spo_index.append(model.spo.clone().data.cpu().numpy())

                np.save(folder_name + '/forward_nb_index', np.array(forward_nb_index))
                np.save(folder_name + '/forward_spo_index', np.array(forward_spo_index))

        print("\n\t TRAIN: ", train_results)

        train_mrr.append(train_results['MRR'])

        hits1310 = train_results['hits@[1,3,10]'].numpy()
        train_hit1.append(hits1310[0])
        train_hit3.append(hits1310[1])
        train_hit10.append(hits1310[2])

        summary_config[train_no]['curr_train_hit10'] = str(hits1310[2])
        summary_config[train_no]['curr_train_mrr'] = str(train_results['MRR'])

        np.save(folder_name + '/loss', np.array(optimizer.loss_list))

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

        max_test_mrr = max(np.array(test_mrr))
        max_test_hits = max(np.array(test_hit10))
        summary_config[train_no]['max_test_hit10'] = str(max_test_hits)
        summary_config[train_no]['max_test_mrr'] = str(max_test_mrr)

        summary_config[train_no]['curr_test_hit10'] = str(hits1310[2])
        summary_config[train_no]['curr_test_mrr'] = str(results['MRR'])

        if max_test_mrr >= float(summary_config['summary']['best_mrr']):
            best_model_flag = 1
            summary_config['summary']['best_train_no'] = str(train_no)
            summary_config['summary']['best_mrr'] = str(max_test_mrr)
            summary_config['summary']['best_hits@10'] = str(max_test_hits)
            torch.save(model.state_dict(), folder_name + '/model_state.pt')

        if (e + 1) % (args.valid * 3) == 0:
            test_g.append(model.g.clone().data.cpu().numpy())
            test_alpha.append(model.alpha.clone().data.cpu().numpy())
            np.save(folder_name + '/test_g', np.array(test_g))
            np.save(folder_name + '/test_alpha', np.array(test_alpha))

            if args.evaluation_mode:
                test_nb_index.append(model.index_array)
                test_spo_index.append(model.spo.clone().data.cpu().numpy())

                np.save(folder_name + '/test_nb_index', np.array(test_nb_index))
                np.save(folder_name + '/test_spo_index', np.array(test_spo_index))


        config['e'] = e
        pickle.dump(config, open(folder_name + '/config.p', 'wb'))

        summary_config[train_no]['e'] = str(e)

        # Update ../results/model/dataset/summary_config.ini file
        with open('../results/{}/{}/summary_config.ini'.format(args.model, args.dataset), 'w') as configfile:
            summary_config.write(configfile)

eps = 0.002
if (max(np.array(test_mrr)) == np.array(test_mrr)[-1]) and (abs(np.array(test_mrr)[-1] - np.array(test_mrr)[-2]) > eps):
    # torch.save(model.state_dict(), folder_name + '/model_state.pt')
    summary_config['summary']['{} not fully trained'.format(train_no)] = 'True'

    with open('../results/{}/{}/summary_config.ini'.format(args.model, args.dataset), 'w') as configfile:
        summary_config.write(configfile)
