
import torch
from torch import optim

from pathlib import Path
import pkg_resources
from kbc.datasets import Dataset
from kbc.models import CP, ComplEx, ConvE
from kbc.regularizers import N3
from kbc.optimizers import KBCOptimizer
from typing import Dict
import pickle

'''
Define Parameters
'''
embedding_size = 200
hw = (10, 20)
kernel_size = (3, 3)
use_bias = True
output_channel = 32
dropouts = (0.3, 0.3, 0.3)
learning_rate = 10
batch_size = 100
max_epochs = 3
valid = 3

#Dataset Class
dataset_name = 'WN18RR'
dataset = Dataset(dataset_name)

#Train example
train_examples = torch.from_numpy(dataset.get_train().astype('int64'))

#Define ConvE model
model = ConvE(dataset.get_shape(), embedding_size, dropouts, use_bias, hw, kernel_size, output_channel)

#Define regularizer
regularizer = N3

device = 'cuda'
model.to(device)

optim_method = optim.Adagrad(model.parameters(), lr=learning_rate)
optimizer = KBCOptimizer(model, regularizer, optim_method, batch_size = batch_size)


def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MRR': m, 'hits@[1,3,10]': h}

# e.g. run
# python kbc/learn.py --dataset 'WN18RR' --model 'ConvE'
#
# cur_loss = 0
# curve = {'train': [], 'valid': [], 'test': []}
# for e in range(max_epochs):
#     cur_loss = optimizer.epoch(train_examples)
#
#     if (e + 1) % valid == 0:
#         valid, test, train = [
#             avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
#             for split in ['valid', 'test', 'train']
#         ]
#
#         curve['valid'].append(valid)
#         curve['test'].append(test)
#         curve['train'].append(train)
#
#         print("\t TRAIN: ", train)
#         print("\t VALID : ", valid)
#
# results = dataset.eval(model, 'test', -1)
# print("\n\nTEST : ", results)

'''
call test dataset
'''

DATA_PATH = Path(pkg_resources.resource_filename('kbc', 'data/'))

root = DATA_PATH / dataset_name

data = {}
for f in ['test', 'valid']:
    in_file = open(str(root / (f + '.pickle')), 'rb')
    data[f] = pickle.load(in_file)





'''
Check predicted labels against the target labels
'''














