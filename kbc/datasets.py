# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from pathlib import Path
import pkg_resources
import pickle
from typing import Dict, Tuple, List

import numpy as np
import torch
from kbc.models import KBCModel
import os

DATA_PATH = Path(pkg_resources.resource_filename('kbc', 'data/'))

class Dataset(object):
    def __init__(self, name: str, use_colab=False):
        self.root = DATA_PATH / name

        self.data = {}

        for f in ['train', 'test', 'valid']:
            if not use_colab:
                in_file = open(str(self.root / (f + '.pickle')), 'rb')
                self.data[f] = pickle.load(in_file)
            else:
                in_file = open('drive/My Drive/SummerProject/Shared/CDT/kbc/kbc/data/' + name + '/' + f + '.pickle',
                               'rb')
                print(in_file)
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
            return pickle.load(open(sorted_file_path, 'rb')), \
                    pickle.load(open(slice_file_path, 'rb'))
        else:
            train = self.get_train()
            train.sort(axis=0)
            i = 0
            curr_ent = train[0, 0]
            slice_dic = {}
            start = 0
            while i < len(train):
                prev_ent = curr_ent
                curr_ent = train[i, 0]

                if prev_ent != curr_ent:
                    slice_dic[prev_ent] = (start, i)
                    start = i

                if i == len(train) - 1:
                    slice_dic[curr_ent] = (start, i + 1)
                i += 1
            pickle.dump(train, open(sorted_file_path, 'wb'))
            pickle.dump(slice_dic, open(slice_file_path, 'wb'))
            return train, slice_dic

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
