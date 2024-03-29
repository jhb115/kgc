from pathlib import Path
import pkg_resources
import pickle
from typing import Dict, Tuple, List

import numpy as np
import torch
from kgc.models import KBCModel
import os

DATA_PATH = '../dataset/raw_data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Dataset(object):
    def __init__(self, name: str, rcp_bool: int = 1):
        self.root = os.path.join(DATA_PATH, name)
        self.name = name
        self.data = {}
        self.rcp_bool = rcp_bool

        for f in ['train', 'test', 'valid']:
            in_file = open(os.path.join(self.root, str(f + '.pickle')), 'rb')
            self.data[f] = pickle.load(in_file)

        maxis = np.max(self.data['train'], axis=0)
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        self.n_predicates = int(maxis[1] + 1)

        if self.rcp_bool:
            self.n_predicates *= 2

        inp_f = open(os.path.join(self.root, 'to_skip.pickle'), 'rb')
        self.to_skip: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
        inp_f.close()

    def get_examples(self, split):
        return self.data[split]

    def get_train(self):
        # .pickle file contains non-reciprocals which is data['train']
        #  this function returns org+reciprocal triplets
        if self.rcp_bool:
            copy = np.copy(self.data['train'])
            tmp = np.copy(copy[:, 0])

            copy[:, 0] = copy[:, 2]
            copy[:, 2] = tmp
            copy[:, 1] += self.n_predicates // 2  # has been multiplied by two.
            return np.vstack((self.data['train'], copy))
        else:
            return self.data['train']

    def get_1hop_nb(self):
        one_hop_path = os.path.join(self.root, 'one_hop_list.npy')
        slice_file_path = os.path.join(self.root, 'one_hop_slice.npy')
        if os.path.exists(one_hop_path) and os.path.exists(slice_file_path):
            print('Sorted train set loaded')
            return np.load(one_hop_path), np.load(slice_file_path)
        else:
            print('Create new sorted list')
            train = self.get_train().astype('int64')

            train = train[train[:, 0].argsort()]  # sorts the dataset in order with respect to subject entity id

            print('Min entity id for {} is {}'.format(self.name, train[0, 0]))

            slice_dic = []
            start = 0
            i = 1
            prev_ent = train[0, 0]
            one_hop_list = []
            candidate_nb = []

            while i < len(train):
                curr_ent = train[i, 0]
                candidate_nb.append(train[i-1, 2])

                if prev_ent != curr_ent:
                    one_hop_list += candidate_nb
                    end = start + len(candidate_nb)
                    slice_dic.append([start, end])
                    candidate_nb = []
                    start = end
                    while prev_ent+1 != curr_ent:
                        slice_dic.append([end, end])
                        prev_ent += 1

                if i == len(train) - 1:
                    candidate_nb.append(train[i, 2])
                    slice_dic.append([start, start + len(candidate_nb)])
                    one_hop_list += candidate_nb

                prev_ent = curr_ent
                i += 1

            one_hop_list = np.array(one_hop_list, dtype=np.int64)
            slice_dic = np.array(slice_dic, dtype=np.int64)

            np.save(one_hop_path, one_hop_list)
            np.save(slice_file_path, slice_dic)

            return one_hop_list, slice_dic

    def get_2hop_nb(self, threshold=20):

        sorted_file_path = os.path.join(self.root, 'two_hop_list.npy')
        slice_file_path = os.path.join(self.root, 'two_hop_slice.npy')
        if os.path.exists(sorted_file_path) and os.path.exists(slice_file_path) and 0:
            # load data if exists
            print('Sorted train set loaded')
            return np.load(sorted_file_path), np.load(slice_file_path)
        else:  # create data if not exists

            one_hop_sorted, one_hop_slice = self.get_1hop_nb()  # sorted-train and slice-dic

            one_hop_sorted.astype('int32')
            one_hop_slice.astype('int32')

            one_hop_sorted = one_hop_sorted.tolist()
            one_hop_slice = one_hop_slice.tolist()

            i = 0
            two_start = 0

            two_hop_list = []
            two_hop_slice = []

            while i < len(one_hop_slice):

                # add one hop neighbors to candidate_nb
                one_start, one_end = one_hop_slice[i]
                curr_one_hop = one_hop_sorted[one_start:one_end]
                two_hop_candidate = curr_one_hop

                # add two hop neighbors to candidate_nb
                for each_obj in list(set(curr_one_hop)):
                    each_start, each_end = one_hop_slice[each_obj]
                    if each_end - each_start < threshold:  # consider only neighbors (two-hop) of nodes neq hubs.
                        two_hop_candidate += one_hop_sorted[each_start:each_end]

                nb_candidates, counts = np.unique(two_hop_candidate, return_counts=True)
                two_hop_candidate = list(nb_candidates[counts > 1]) + list(nb_candidates)

                two_end = two_start + len(two_hop_candidate)
                two_hop_slice.append([two_start, two_end])
                two_hop_list += two_hop_candidate
                two_start = two_end

            two_hop_list = np.array(two_hop_list, dtype=np.int32)
            two_hop_slice = np.array(two_hop_slice, dtype=np.int32)

            np.save(sorted_file_path, two_hop_list)
            np.save(slice_file_path, two_hop_slice)

            return two_hop_list, two_hop_slice

    def eval(
            self, model: KBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both',
            at: Tuple[int] = (1, 3, 10)
    ):
        test = self.get_examples(split)
        examples = torch.from_numpy(test.astype('int64')).to(device)
        missing = [missing_eval]
        if missing_eval == 'both':
            missing = ['rhs', 'lhs']

        if not self.rcp_bool:
            missing = ['rhs']

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

            #  where we run the model on self.to_skip
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
