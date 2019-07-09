import argparse
from typing import Dict
import pickle
import configparser

import torch
from torch import optim

from kbc.datasets import Dataset
from kbc.models import CP, ComplEx, ConvE
from kbc.regularizers import N2, N3
from kbc.optimizers import KBCOptimizer
import os
import numpy as np

#For reproducilibility
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()

datasets = ['FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10']
parser.add_argument('--dataset', choices=datasets)

models = ['CP', 'ComplEx', 'ConvE']
parser.add_argument('--model', choices=models)

parser.add_argument('--train_no')

args = parser.parse_args()

folder_path = './results/{}/{}/train{}'.format(args.model, args.dataset, str(int(args.train_no)))

# check if the folder exists
if not os.path.exists(folder_path):
    raise Exception('You do not have folder named:{}'.format(folder_path))

# folder path format 'results/ComplEx/FB15K/train1'
# Get the configuration
config = pickle.load(open(folder_path + '/config.p', 'rb'))

# Get Dataset
dataset = Dataset(config['dataset'])
examples = torch.from_numpy(dataset.get_train().astype('int64'))

rank, init = [int(config['rank']), float(config['init'])]

print(dataset.get_shape())

model = {
    'CP': lambda: CP(dataset.get_shape(), config['rank'], config['init']),
    'ComplEx': lambda: ComplEx(dataset.get_shape(), config['rank'], config['init']),
    'ConvE': lambda: ConvE(dataset.get_shape(), config['rank'], config['dropouts'], config['use_bias'],
                           config['hw'], config['kernel_size'], config['output_channel'])
}[config['model']]()

regularizer = {
    'N0': 'N0',
    'N2': N2(config['reg']),
    'N3': N3(config['reg']),
}[config['regularizer']]

device = 'cuda'
model.to(device)

if config['model'] == "ConvE":
    model.init()

optim_method = {
    'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=config['learning_rate']),
    'Adam': lambda: optim.Adam(model.parameters(), lr=config['learning_rate'], betas=(config['decay1'], config['decay2'])),
    'SGD': lambda: optim.SGD(model.parameters(), lr=config['learning_rate'])
}[config['optimizer']]()

optimizer = KBCOptimizer(model, regularizer, optim_method, config['batch_size'], loss_type=config['loss'])

model.load_state_dict(torch.load(folder_path + '/model_state.pt'))

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
train_i = 0
test_i = 0

split_name = ['train', 'valid']
hits_name = ['_hits@1', '_hits@3', '_hits@10']

train_mrr = list(np.load(folder_path + '/train_mrr.npy'))
train_hit1 = list(np.load(folder_path + '/train_hit1.npy'))
train_hit3 = list(np.load(folder_path + '/train_hit3.npy'))
train_hit10 = list(np.load(folder_path + '/train_hit10.npy'))

valid_mrr = list(np.load(folder_path + '/train_mrr.npy'))
valid_hit1 = list(np.load(folder_path + '/train_hit1.npy'))
valid_hit3 = list(np.load(folder_path + '/train_hit3.npy'))
valid_hit10 = list(np.load(folder_path + '/train_hit10.npy'))


test_mrr = list(np.load(folder_path + '/train_mrr.npy'))
test_hit1 = list(np.load(folder_path + '/train_hit1.npy'))
test_hit3 = list(np.load(folder_path + '/train_hit3.npy'))
test_hit10 = list(np.load(folder_path + '/train_hit10.npy'))

# Save the configuration file and txt file description.
# What to include in the configuration file and txt file:
# Config: args.model, args.dataset,
# config['max_epochs'], e, config['regularizer'], config['optimizer'], config['rank'], config['batch_size'],
# config['reg'], config['init'], config['learning_rate']
# if args.model == 'ConvE':
# args.dropouts, config['use_bias'], args.kernel_size, config['output_channel'], config['hw']

config_ini = configparser.ConfigParser()
config_ini.read(folder_path + '/config.ini')

for e in range(config['e']+1, config['max_epochs']):
    print('\n train epoch = ', e+1)
    cur_loss = optimizer.epoch(examples)

    if (e + 1) % config['valid'] == 0 or (e+1) == config['max_epochs']:

        torch.save(model.state_dict(), folder_path + '/model_state.pt')

        train_results, valid_results = [
            avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
            for split in split_name
        ]

        print("\n\t TRAIN: ", train_results)
        print("\t VALID : ", valid_results)

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

        np.save(folder_path + '/train_mrr', np.array(train_mrr))
        np.save(folder_path + '/train_hit1', np.array(train_hit1))
        np.save(folder_path + '/train_hit3', np.array(train_hit3))
        np.save(folder_path + '/train_hit10', np.array(train_hit10))

        np.save(folder_path + '/valid_mrr', np.array(valid_mrr))
        np.save(folder_path + '/valid_hit1', np.array(valid_hit1))
        np.save(folder_path + '/valid_hit3', np.array(valid_hit3))
        np.save(folder_path + '/valid_hit10', np.array(valid_hit10))

#    if (e+1) % config['valid'] == 0 or (e+1) == config['max_epochs']:

        results = avg_both(*dataset.eval(model, 'test', -1))

        test_mrr.append(results['MRR'])

        hits1310 = results['hits@[1,3,10]'].numpy()

        test_hit1.append(hits1310[0])
        test_hit3.append(hits1310[1])
        test_hit10.append(hits1310[2])

        print("\n\nTEST : ", results)

        np.save(folder_path + '/test_mrr', np.array(test_mrr))
        np.save(folder_path + '/test_hit1', np.array(test_hit1))
        np.save(folder_path + '/test_hit3', np.array(test_hit3))
        np.save(folder_path + '/test_hit10', np.array(test_hit10))

        pickle.dump(config, open(folder_path + '/config.p', 'wb'))

        config_ini['setup']['e'] = str(e)

        with open(folder_path + '/config.ini', 'w') as configfile:
            config_ini.write(configfile)

