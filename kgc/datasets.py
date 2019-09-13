from pathlib import Path
import pkg_resources
import pickle
from typing import Dict, Tuple, List

import numpy as np
import torch
from kgc.models import KBCModel
import os

DATA_PATH = Path(pkg_resources.resource_filename('kgc', 'data/'))

class Dataset(object):
    def __init__(self, name: str):
        self.root = DATA_PATH / name
        self.name = name
        self.data = {}

        for f in ['train', 'test', 'valid']:
            in_file = open(str(self.root / (f + '.pickle')), 'rb')
            self.data[f] = pickle.load(in_file)

        maxis = np.max(self.data['train'], axis=0)
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        self.n_predicates = int(maxis[1] + 1)
        self.n_predicates *= 2

        inp_f = open(str(self.root / f'to_skip.pickle'), 'rb')
        self.to_skip: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
        inp_f.close()

    def get_examples(self, split):
        return self.data[split]

    def get_train(self):
        # .pickle file contains non-reciprocals which is data['train']
        #  this function returns org+reciprocal triplets
        copy = np.copy(self.data['train'])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2  # has been multiplied by two.
        return np.vstack((self.data['train'], copy))

    def get_sorted_train(self):
        sorted_file_path = self.root / 'sorted_train.pickle'
        slice_file_path = self.root / 'slice_train.pickle'
        if os.path.exists(sorted_file_path) and os.path.exists(slice_file_path):
            # load data if exists
            print('Sorted train set loaded')
            return pickle.load(open(sorted_file_path, 'rb')), \
                    pickle.load(open(slice_file_path, 'rb'))
        else:  # create data if not exists
            print('Create new sorted list')
            train = self.get_train().astype('int64')

            train = train[train[:, 0].argsort()]  # sorts the dataset in order with respect to subject entity id
            i = 0
            curr_ent = train[0, 0]

            print('Min entity id for {} is {}'.format(self.name, curr_ent))

            ent_idx_list = list(range(curr_ent, max(train[:, 0]) + 1 + curr_ent))

            slice_dic = []
            start = 0
            while i < len(train):
                prev_ent = curr_ent
                ent_idx = ent_idx_list[len(slice_dic)]
                curr_ent = train[i, 0]

                if prev_ent != curr_ent:
                    while ent_idx_list[len(slice_dic) + 1] != curr_ent:
                        slice_dic.append([ent_idx_list[len(slice_dic) + 1], start, start, 0])
                    slice_dic.append([prev_ent, start, i, i - start])  # slice_dic[i] == (subject, start, end, degree)
                    start = i
                    ent_idx += 1

                if i == len(train) - 1:
                    slice_dic.append([curr_ent, start, i+1, i+1 - start])

                i += 1

            slice_dic = np.array(slice_dic, dtype=np.int64)
            slice_dic = slice_dic[slice_dic[:, 0].argsort()]

            nb_degrees = slice_dic[train[:, 2], 3]

            i_train = np.lexsort((nb_degrees, train[:, 0]))
            # sort in terms of degrees of neighbouring nodes first then sort with respect to train id
            train = train[i_train]
            slice_dic = slice_dic[:, :3]

            pickle.dump(train, open(sorted_file_path, 'wb'))
            pickle.dump(slice_dic, open(slice_file_path, 'wb'))
            return train, slice_dic

    def get_2hop_nb(self):
        '''
        2-hop neighborhood
        :return: 2_hop_sorted_train, 2_hop_slice_train
        '''
        sorted_file_path = self.root / 'two_hop_sorted.pickle'
        slice_file_path = self.root / 'two_hop_slice.pickle'
        if os.path.exists(sorted_file_path) and os.path.exists(slice_file_path):
            # load data if exists
            print('Sorted train set loaded')
            return pickle.load(open(sorted_file_path, 'rb')), \
                   pickle.load(open(slice_file_path, 'rb'))
        else:  # create data if not exists
            print('Create new sorted list')
            one_hop_sorted, one_hop_slice = self.get_sorted_train()  # sorted-train and slice-dic
            # one_hop_sorted = [ [subj, rel, obj] ]
            # one_hop_slice = [ [subj, start, end, length] ]

            i = 0
            two_start = 0
            two_end = 0

            # two_hop_sorted = [ obj1, obj2, obj3, ..., objN ]
            # two_hop_slice = [ [start, end] ... ]

            two_hop_sorted = []
            two_hop_slice = []

            while i < len(one_hop_slice):

                # add one hop neighbors to candidate_nb
                _, one_start, one_end = one_hop_slice[i]
                curr_one_hop_nb = one_hop_sorted[one_start:one_end, 2]
                two_hop_sorted += curr_one_hop_nb
                two_end += len(curr_one_hop_nb)

                #add two hop neighbors to candidate_nb
                for each_obj in curr_one_hop_nb:
                    _, each_start, each_end = one_hop_slice[each_obj]
                    two_hop_sorted += one_hop_sorted[each_start:each_end, 2]
                    two_end += each_end - each_start

                two_hop_slice.append([two_start, two_end])
                two_start = two_end
                i += 1

            pickle.dump(two_hop_sorted, open(sorted_file_path, 'rb'))
            pickle.dump(two_hop_slice, open(slice_file_path, 'rb'))

            return two_hop_sorted, two_hop_slice

    def eval(
            self, model: KBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both',
            at: Tuple[int] = (1, 3, 10)
    ):
        test = self.get_examples(split)
        examples = torch.from_numpy(test.astype('int64')).cuda()
        missing = [missing_eval]
        if missing_eval == 'both':
            missing = ['rhs', 'lhs']

        mean_reciprocal_rank = {}
        hits_at = {}

        for m in missing:
            q = examples.clone()
            if n_queries > 0:
                permutation = torch.randperm(len(examples))[:n_queries]
                q = examples[permutation]
            if m == 'lhs':
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.n_predicates // 2
            ranks = model.get_ranking(q, self.to_skip[m], batch_size=500)
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                at
            ))))

        return mean_reciprocal_rank, hits_at

    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities


# data_list = ['FB15K', 'FB237', 'WN', 'WN18RR', 'YAGO3-10']
#
#     for each_data in data_list:
#         mydata = Dataset(each_data)
#         sorted_train, slice_dic = mydata.get_sorted_train()
#         unsorted_train = mydata.get_train()
#
#         for i in slice_dic:
#             if i[1] == np.nan:
#                 print('nan exists in {}'.format(mydata.name))
#
# '''
# FB15K: min entity = 0, all entity has neighbor
# FB237: min entity = 0, some entity does not have neighbour (i.e. entity not observed in train set)
# WN: min entity = 0, all entity has neighbor
# WN18RR: min entity = 0, some entity does not have neighbour
# YAGO3-10: min entity = 0, some entity does not have neighbour
# '''
