from pathlib import Path
import pkg_resources
import pickle
from kbc.datasets import Dataset
import matplotlib.pyplot as plt


#%%%%
'''
class Dataset
get_train -> returns train examples (examples include reciprocal triplets)
'''

data_list = ['FB15K', 'FB237', 'WN', 'WN18RR', 'YAGO3-10']
file_type = ['train', 'valid', 'test']

for each_data in data_list:

    print('\n{}'.format(each_data))
    mydata = Dataset(each_data, use_colab = False)

    train = mydata.get_train()
    org_train = mydata.data['train']

    # Check if all reciprocal relationship exist in train
    # Yes
    print('Compare 2 x number of org triplets vs. number of org+reciprocal triplets')
    if (len(org_train)*2) == len(train):
        print('Equal')
    else:
        print('Not equal')

    print('\nCompare number of relations vs. max(id of rel) + 1')
    if mydata.n_predicates == max(train[:, 1])+1:
        print('Equal')
    else:
        print('Not Equal')

'''
They are all Equal
'''

#%%%
# Check if all entities in valid and test exist in train
# FB237, WN18RR, YAGO3-10 HAVE UNOBSERVED ENTITIES IN TRAIN AND VALID SET

DATA_PATH = Path(pkg_resources.resource_filename('kbc', 'data/'))
print(DATA_PATH)

data_list = ['FB15K', 'FB237', 'WN', 'WN18RR', 'YAGO3-10']
file_type = ['train', 'valid', 'test']

for each_data in data_list:
    print('\n{}'.format(each_data))
    root = DATA_PATH / each_data

    file_dic = {}
    for each_file in file_type:
        in_file = open(str((root / (each_file + '.pickle'))), 'rb')
        file_dic[each_file] = pickle.load(in_file)

    train_entity_set = set(file_dic['train'][:, 0]).union( set(file_dic['train'][:, 2]))
    valid_entity_set = set(file_dic['valid'][:, 0]).union(set(file_dic['valid'][:, 2]))
    test_entity_set = set(file_dic['test'][:, 0]).union(set(file_dic['test'][:, 2]))

    tst_trn = test_entity_set.difference(train_entity_set)
    vld_trn = valid_entity_set.difference(train_entity_set)

    print('{} number of test entities not found in train set'.format(len(tst_trn)))
    print(tst_trn)
    print('{} number of valid entities not found in train set'.format(len(vld_trn)))
    print(vld_trn)

'''

FB15K
0 number of test entities not found in train set
set()
0 number of valid entities not found in train set
set()

FB237
29 number of test entities not found in train set
{11909, 12679, 9865, 7696, 3478, 7195, 7580, 10782, 14498, 10020, 12718, 9008, 5042, 3896, 4537, 186, 14140, 6723, 4553, 2894, 10190, 7123, 7381, 7258, 11363, 11626, 7664, 3192, 7548}
8 number of valid entities not found in train set
{12868, 2894, 12143, 3440, 5650, 1975, 6712, 473}

WN
0 number of test entities not found in train set
set()
0 number of valid entities not found in train set
set()

WN18RR
209 number of test entities not found in train set
{38400, 5, 14856, 1033, 23049, 6672, 17427, 9235, 38933, 26135, 18456, 37926, 31787, 9772, 7725, 25140, 23093, 8756, 10293, 12861, 31808, 10307, 37444, 23109, 14404, 25675, 9292, 10315, 12369, 25172, 25690, 19036, 93, 23141, 11880, 18026, 22122, 8814, 6772, 38006, 33913, 40059, 25217, 25729, 4739, 35460, 32388, 36998, 2183, 39566, 32914, 21657, 25753, 27801, 12445, 8863, 11935, 30883, 27302, 38057, 17066, 17067, 22698, 18605, 23723, 12462, 32944, 18608, 26289, 12464, 18100, 36023, 6329, 12474, 33979, 20158, 27841, 25282, 7881, 6348, 37071, 28879, 8404, 34519, 36057, 30426, 14558, 34530, 35044, 33510, 10985, 23786, 33004, 36589, 33519, 25330, 25848, 4350, 23294, 9473, 33026, 11521, 15618, 25086, 39183, 15631, 9491, 35092, 20757, 9494, 5402, 36635, 40219, 286, 24862, 7467, 9516, 16171, 36654, 38192, 12594, 32562, 12596, 27446, 7479, 24375, 27447, 7998, 25415, 330, 33611, 11595, 23887, 33104, 15696, 6995, 345, 11098, 17243, 12122, 22877, 33121, 7521, 867, 4456, 17257, 18793, 3946, 33646, 31086, 7536, 9584, 33652, 26484, 32630, 31095, 15737, 9596, 26497, 14722, 31618, 24964, 13188, 9608, 12682, 18831, 38287, 30607, 23445, 8085, 14230, 24472, 35228, 36255, 8098, 7078, 9127, 25518, 9653, 2486, 30135, 8633, 16830, 38336, 7105, 1474, 27588, 29124, 26059, 33741, 13774, 33233, 26065, 18388, 10198, 8152, 8666, 38363, 35805, 39903, 34791, 23528, 6636, 19438, 6137, 36858, 30203, 6654, 38399}
198 number of valid entities not found in train set
{10753, 23554, 19462, 18441, 31241, 19468, 19470, 16401, 25618, 38420, 23060, 8729, 18971, 17407, 3613, 31772, 13856, 12322, 37924, 1578, 25132, 9772, 25135, 1584, 34353, 17973, 20535, 31801, 570, 30267, 11324, 24125, 31809, 10307, 24647, 24139, 2124, 7245, 23118, 29264, 32339, 26709, 16982, 11861, 20059, 22116, 38500, 14948, 18026, 25198, 32366, 25712, 25201, 19063, 31351, 33401, 23677, 25215, 28288, 33922, 33931, 9869, 29332, 25238, 39577, 27802, 30879, 4256, 24230, 10410, 12461, 28334, 32944, 10417, 31921, 18100, 14518, 36023, 10936, 32438, 16057, 33979, 31420, 5824, 11458, 22723, 25795, 36039, 24264, 14538, 24267, 10443, 17616, 33493, 33495, 10455, 28376, 8410, 15064, 23772, 23775, 24290, 27876, 12518, 5865, 26351, 10992, 12015, 25333, 33017, 25347, 30982, 21257, 2314, 25359, 40727, 39704, 17692, 39197, 286, 39708, 38176, 25376, 13085, 30500, 27941, 6443, 25387, 5428, 31033, 10043, 36671, 27980, 13135, 24405, 33622, 37719, 33112, 345, 31064, 26460, 22878, 25443, 4456, 18793, 36716, 15730, 26484, 2933, 12149, 39799, 24440, 7033, 19837, 26497, 28545, 17800, 9608, 11146, 33163, 17293, 18831, 9616, 4499, 22419, 10646, 28576, 7078, 22452, 9653, 10679, 30135, 32186, 23484, 16829, 26047, 38336, 7105, 15296, 23491, 36804, 27588, 32199, 9162, 7118, 15824, 13777, 17879, 24023, 39903, 27624, 17387, 23533, 37870, 25583, 14831, 8692, 38399}
YAGO3-10


18 number of test entities not found in train set
{66016, 65538, 91426, 6180, 83269, 20038, 40165, 44387, 32809, 35306, 56651, 59468, 29837, 82030, 18165, 48291, 63298, 30463}
22 number of valid entities not found in train set
{60420, 109958, 100872, 47119, 63887, 66210, 48291, 27954, 36148, 47928, 94663, 1992, 11465, 101327, 81874, 33619, 112216, 101734, 119019, 35821, 23662, 51070}

'''

#%%%
# Check if all relations in valid and test exist in train
# All relationships in train set found in valid and test set
DATA_PATH = Path(pkg_resources.resource_filename('kbc', 'data/'))
print(DATA_PATH)

data_list = ['FB15K', 'FB237', 'WN', 'WN18RR', 'YAGO3-10']
file_type = ['train', 'valid', 'test']

for each_data in data_list:
    print('\n{}'.format(each_data))
    root = DATA_PATH / each_data

    file_dic = {}
    for each_file in file_type:
        in_file = open(str((root / (each_file + '.pickle'))), 'rb')
        file_dic[each_file] = pickle.load(in_file)

    train_rel_set = set(file_dic['train'][:, 1])
    valid_rel_set = set(file_dic['valid'][:, 1])
    test_rel_set = set(file_dic['test'][:, 1])

    tst_trn = test_rel_set.difference(train_rel_set)
    vld_trn = valid_rel_set.difference(train_rel_set)

    print('{} number of test rel not found in train set'.format(len(tst_trn)))
    print(tst_trn)
    print('{} number of valid rel not found in train set'.format(len(vld_trn)))
    print(vld_trn)

#%%%
'''
Check if valid and test set also have reciprocals
'''

from kbc.datasets import Dataset

data_list = ['FB15K', 'FB237', 'WN', 'WN18RR', 'YAGO3-10']
file_type = ['train', 'valid', 'test']

for each_data in data_list:

    print('\n{}'.format(each_data))

    root = DATA_PATH / each_data
    mydata = Dataset(each_data, use_colab=False)

    file_dic = {}
    train = mydata.get_train()
    org_train = mydata.data['train']

    # Check if all reciprocal relationship exist in train
    # Yes
    print('Compare 2 x number of org triplets vs. number of org+reciprocal triplets')
    if (len(org_train)*2) == len(train):
        print('Equal')
    else:
        print('Not equal')

    print('\nCompare number of relations vs. max(id of rel) + 1')
    if mydata.n_predicates == max(train[:, 1])+1:
        print('Equal')
    else:
        print('Not Equal')

#%%%
'''
Explore statistics of N_nb (number of neighbours for a given subject)
'''

# Consider only the train set (org + reciprocal)
data_list = ['FB15K', 'FB237', 'WN', 'WN18RR', 'YAGO3-10']
plt.close('all')
for each_data in data_list:

    mydata = Dataset(each_data, use_colab=False)
    sorted_train, slice_dic = mydata.get_sorted_train()

    # Iterate through slice_dic
    diff_list = [0] * len(slice_dic)
    for i, each_slice in enumerate(slice_dic.values()):
        diff_list[i] = each_slice[1] - each_slice[0]

    plt.figure()
    plt.hist(diff_list, bins=500, range=(min(diff_list), 300))
    plt.title(each_data)

    print('')
    print(each_data)
    print('Train Data Length = ', len(sorted_train))
plt.show()

'''
FB15K
Train Data Length =  966284
FB237
Train Data Length =  544230
WN
Train Data Length =  282884
WN18RR
Train Data Length =  173670

suggested max_NB:
FB15K = 50 ~ 100
FB237 = 50 ~ 100
WN = 20
WN18RR = 20
YAGO3-10 = 40 ~ 50
'''
#%%%
# Check if score for Context_CP works correctly

from abc import ABC, abstractmethod
from typing import Tuple
import torch
from torch import nn
import numpy as np


class Context_CP(nn.Module):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3, data_name: str = 'FB15K', sorted_data = None,
            slice_dic = None, max_NB = 50
    ):
        super(Context_CP, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.data_name = data_name

        self.lhs = nn.Embedding(sizes[0], rank, sparse=True)
        self.rel = nn.Embedding(sizes[1], rank, sparse=True)
        self.rhs = nn.Embedding(sizes[2], rank, sparse=True)

        self.lhs.weight.data *= init_size
        self.rel.weight.data *= init_size
        self.rhs.weight.data *= init_size

        # Context related parameters
        self.W = nn.Linear(int(3*rank), rank)  # W for w = [lhs; rel; rhs]^T W
        self.sorted_data = sorted_data
        self.slice_dic = slice_dic
        self.max_NB = max_NB

    def score(self, x):
        # Need to change this
        # tot_score = local_score + context_score

        tensor_x = torch.from_numpy(x.astype('int64'))

        self.chunk_size = tensor_x.size()[0]

        lhs = self.lhs(tensor_x[:, 0])  # shape == (chunk_size, k)
        rel = self.rel(tensor_x[:, 1])
        rhs = self.rhs(tensor_x[:, 2])

        # concatenation of lhs, rel, rhs
        trp_E = torch.cat((lhs, rel, rhs), dim=1) # trp_E.shape == (chunk_size, 3k)

        # Get attention weight vector, where W.shape == (3k, k)
        w = self.W(trp_E) # w.shape == (chunk_size, k)

        # Get nb_E = [ nb_E_o ]
        nb_E = self.get_neighbor(x[:, 0])  # nb_E.shape == (chunk_size, max_NB, k)
        alpha = torch.softmax(torch.einsum('bk,bmk->bm', w, nb_E), dim = 1)  # matrix multiplication of w^T nb_E
        # matrix multiplication inside gives (chunk_size x max_NB)
        # output shape is identical

        # Get context vector
        e_c = torch.einsum('bm,bmk->bk', alpha, nb_E)  # (chunk_size, k)

        # Get tot_score
        tot_score = torch.sum(lhs * rel * rhs * e_c, 1, keepdim=True)

        return tot_score

    def get_neighbor(self, subj_list):
        # Need to find a way to index pytorch tensor

        # return neighbor (N_subject, N_nb_max, k)
        nb_E = torch.zeros(self.chunk_size, self.max_NB, self.rank)
        # shape == (batch_size, max_NB, emb_size)

        # Need to consider how to avoid this for loop
        for i, each_subj in enumerate(subj_list):
            if each_subj in self.slice_dic:  # since the subject entity in train set may not be present in valid/test set
                start_i, end_i = self.slice_dic[each_subj]
                length = end_i - start_i
                nb_list = torch.from_numpy(self.sorted_data[start_i: end_i, 2].astype('int64'))  # ignore relation for now

                if self.max_NB > length:  # pad with zeros
                    nb_E[i, :length, :] = self.rhs(nb_list[:])
                else:  # truncate
                    nb_E[i, :, :] = self.rhs(nb_list[:self.max_NB])

            # check self.rhs.shape == (self.max_NB, rank)

        return nb_E  # shape == (chunk_size, self.max_NB, rank), yes

    def forward(self, x):
        # Need to change this
        tensor_x = torch.from_numpy(x.astype('int64'))
        self.chunk_size = len(x)

        lhs = self.lhs(tensor_x[:, 0])
        rel = self.rel(tensor_x[:, 1])
        rhs = self.rhs(tensor_x[:, 2])

        # concatenation of lhs, rel, rhs
        trp_E = torch.cat((lhs, rel, rhs), dim=1)  # trp_E.shape == (chunk_size, 3k)

        # Get attention weight vector, where W.shape == (3k, k)
        w = self.W(trp_E)  # w.shape == (chunk_size, k)

        # Get nb_E
        nb_E = self.get_neighbor(x[:, 0])  # nb_E.shape == (chunk_size, max_NB, k)
        alpha = torch.softmax(torch.einsum('bk,bmk->bm', w, nb_E), dim=1)
        # matrix multiplication inside gives (chunk_size x max_NB)
        # alpha.shape == (chunk_size, max_NB)

        # Get context vector
        e_c = torch.einsum('bm,bmk->bk', alpha, nb_E)  # (chunk_size, k)

        # Get tot_score
        tot_forward = (lhs * rel * e_c) @ self.rhs.weight.t()

        return tot_forward, (lhs, rel, rhs)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        return self.lhs(queries[:, 0]).data * self.rel(queries[:, 1]).data

#%%%
#Test

from kbc.datasets import Dataset

mydata = Dataset('FB15K', use_colab=False)
sorted_data, slice_dic = mydata.get_sorted_train()

chunk_list = list(set(np.random.randint(low = 2, high = 50, size = 50)))
print(chunk_list)
print(len(chunk_list))

x = mydata.get_train()[chunk_list]  # exemplary query
x_tensor = torch.from_numpy(x.astype('int64'))

#%%%
'''
Check save and load functionality of PyTorch
'''

class my_model(nn.Module):
    def __init__(self, sizes=(300,20,300), rank=20):
        super(my_model, self).__init__()

        self.lhs = nn.Embedding(sizes[0], rank, sparse=True)
        self.rel = nn.Embedding(sizes[1], rank, sparse=True)
        self.rhs = nn.Embedding(sizes[2], rank, sparse=True)

test_CP = my_model()

torch.save(test_CP.lhs.state_dict(), './kbc/scrap_data/lhs.pt')

#%%%
mymodel2 = my_model()
mymodel2.lhs.load_state_dict(torch.load('./kbc/scrap_data/lhs.pt'))
#%%%