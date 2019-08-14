# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import tqdm
import torch
from torch import nn
from torch import optim

from kbc.models import KBCModel
from kbc.regularizers import Regularizer

# include pre-training functionality

class KBCOptimizer(object):
    def __init__(
            self, model: KBCModel, regularizer: Regularizer, optimizer: optim.Optimizer, batch_size: int = 256,
            verbose: bool = True, n_freeze: int = 0
    ):
        self.model = model
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.n_freeze = n_freeze

        if self.n_freeze > 0:
            self.freeze_flag = 1
            # freeze the embedding of original embedding
            # self.model.embeddings[0], self.model.embeddings[1], self.model.embeddings[2], self.model.embeddings[3]
        else:
            self.freeze_flag = 0

        self.model_name = 'Context_ComplEx'

    # need to define optimizer with frozen embeddings at the start (let require grad = False, filter out in the parameter)
    # then we need to add this parameter by optim

    def epoch(self, examples: torch.LongTensor):

        if self.freeze_flag == 0 and self.n_freeze > 0:
            if self.model_name == 'Context_CP':
                self.model.lhs.weight.requires_grad = True
                self.model.rel.weight.requires_grad = True
                self.model.rhs.weight.requires_grad = True

                self.optimizer.add_param_group({'lhs': self.model.lhs.parameter(),
                                                'rhs': self.model.rhs.parameter(),
                                                'rel': self.model.rel.parameter()})
            elif self.model_name == 'Context_ComplEx':
                for i in range(2):
                    self.model.embeddings[i].weight.requires_grad = True
                self.optimizer.add_param_group({'embeddings': self.model.embeddings.parameters()})

            self.freeze_flag = None

        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean')

        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin:b_begin + self.batch_size
                ].cuda()

                predictions, factors = self.model.forward(input_batch)
                truth = input_batch[:, 2]

                l_fit = loss(predictions, truth)
                if self.regularizer == 'N0':
                    l = l_fit
                elif self.model.context_flag:
                    l_reg = self.regularizer.forward(factors, g=self.model.g)
                    l = l_fit + l_reg
                else:
                    l_reg = self.regularizer.forward(factors)
                    l = l_fit + l_reg

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.0f}')
