# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_

class KBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    q = self.get_queries(these_queries)

                    scores = q @ rhs
                    targets = self.score(these_queries)

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, np.array(filter_in_chunk, dtype = np.int64)] = -1e6

                        else:
                            scores[i, np.array(filter_out, dtype=np.int64)] = -1e6

                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores > targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks

# Customizable parameters for ConvE:
# dropouts ( = (0.3,0.3,0.3) )
# kernel_size ( = (3,3) )
# output_channel ( = 32 )
# rank ( = embedding_dim = 200)
# HW = (10, 20)


# This is CP model that combines two components together:
# local (individual triplet) + global (context of neighbourhood)

# suggested max_NB:
# FB15K = 50 ~ 100
# FB237 = 50 ~ 100
# WN = 20
# WN18RR = 20
# YAGO3-10 = 40 ~ 50

class Context_CP(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int, sorted_data: np.ndarray,
            slice_dic: np.ndarray, max_NB: int = 50, init_size: float = 1e-3,
            data_name: str = 'FB15K'
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

    def score(self, x: torch.Tensor):

        self.chunk_size = len(x)

        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

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
        tot_score = torch.sum(lhs * rel * rhs * e_c, 1, keepdim=True)

        return tot_score

    def get_neighbor(self, subj: np.ndarray):
        # return neighbor (N_subject, N_nb_max, k)

        index_array = np.zeros(shape=(len(subj), self.max_NB), dtype=np.int32)

        for i, each_subj in enumerate(subj):
            _, start_i, end_i = self.slice_dic[each_subj]
            length = end_i - start_i

            if length > 0:
                if self.max_NB > length:
                    index_array[i, :length] = self.sorted_data[start_i:end_i, 2]
                else:
                    index_array[i, :] = self.sorted_data[start_i:start_i+self.max_NB, 2]

        # Convert index_array into a long tensor for indexing the embedding.
        index_tensor = torch.LongTensor(index_array).cuda()

        return self.rhs(index_tensor)

    def forward(self, x: torch.Tensor):
        # Need to change this

        self.chunk_size = len(x)

        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

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

    def get_queries(self, x: torch.Tensor):  # need to include context part
        # x is a numpy array (equivalent to queries)

        self.chunk_size = len(x)

        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

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

        return lhs.data * rel.data * e_c

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)


class ConvE(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            dropouts: Tuple[float, float, float] = (0.3, 0.3, 0.3),
            use_bias: bool=True, hw: Tuple[int, int] = (0, 0), kernel_size: Tuple[int, int] = (3, 3), output_channel=32
    ):
        super(ConvE, self).__init__()
        self.sizes = sizes
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.embedding_dim = rank  # For ConvE, we shall refer rank as the embedding dimension
        self.use_bias = use_bias
        self.dropouts = dropouts  # (input_dropout, dropout, feature_map_dropout)

        self.H_dim, self.W_dim = self.image_dim(hw)

        num_e = max(sizes[0], sizes[2])

        self.emb_e = nn.Embedding(num_e, self.embedding_dim, padding_idx=0)  # equivalent to both lhs and rhs
        self.emb_rel = nn.Embedding(sizes[1], self.embedding_dim, padding_idx=0)
        self.inp_drop = nn.Dropout(self.dropouts[0])

        self.hidden_drop = nn.Dropout(self.dropouts[1])
        self.feature_map_drop = nn.Dropout2d(self.dropouts[2])

        self.conv1 = nn.Conv2d(1, self.output_channel, self.kernel_size, 1, 0, bias=self.use_bias)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(self.output_channel)
        self.bn2 = nn.BatchNorm1d(self.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_e)))

        fc_dim = self.linear_dim()
        self.fc = nn.Linear(fc_dim, self.embedding_dim)

    def linear_dim(self):  # calculates the dimension of fc layer in ConvE
        fc_dim = self.output_channel * (self.H_dim * 2 - self.kernel_size[0] + 1) * (self.W_dim - self.kernel_size[1] + 1)
        return fc_dim

    def image_dim(self, hw):
        if hw == (0, 0):  # -> find rectangular H, W for reshaping
            w = np.sqrt(self.embedding_dim*2)  # due to the vertical stacking
            h = w/2
            if int(h) != h or int(w) != w:
                raise ValueError('H and W are not integer')
        else:
            h, w = hw
            if h * w != self.embedding_dim:
                raise ValueError('H x W must be equal to the rank')

        return int(h), int(w)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    # Work on score and forward
    def score(self, x):
        lhs = self.emb_e(x[:, 0])
        rel = self.emb_rel(x[:, 1])
        rhs = self.emb_e(x[:, 2])

        batch_size = len(x)

        e1_embedded = lhs.view(-1, 1, self.H_dim, self.W_dim)
        rel_embedded = rel.view(-1, 1, self.H_dim, self.W_dim)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        y = self.inp_drop(stacked_inputs)
        y = self.conv1(y)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.feature_map_drop(y)  # vec(f[e_s; rel] * w)
        y = y.view(batch_size, -1)
        y = self.fc(y)  # vec(f[e_s;rel] * w) W
        y = self.hidden_drop(y)
        y = self.bn2(y)
        y = F.relu(y)  # f(vec(f[e_s; rel] * w) W
        y = y * rhs  # f(vec(f[e_s; rel] * w) W) e_o

        #y = torch.sigmoid(y)  # p = sigmoid( psi_r (e_s, e_o) )

        return torch.sum(y, 1, keepdim=True)

    def forward(self, x):
        lhs = self.emb_e(x[:, 0])
        rel = self.emb_rel(x[:, 1])
        rhs = self.emb_e(x[:, 2])

        batch_size = len(x)

        e1_embedded = lhs.view(-1, 1, self.H_dim, self.W_dim)
        rel_embedded = rel.view(-1, 1, self.H_dim, self.W_dim)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)  # [e_s; rel]
        y = self.inp_drop(stacked_inputs)
        y = self.conv1(y)  # [e_s; rel] * w
        y = self.bn1(y)
        y = F.relu(y)  # f([e_s; rel] * w
        y = self.feature_map_drop(y)  # vec( f([e_s;rel]) )
        y = y.view(batch_size, -1)
        y = self.fc(y)  # vec( f([e_s;rel]) ) W
        y = self.hidden_drop(y)
        y = self.bn2(y)
        y = F.relu(y)  # f( vec( f([e_s;rel]) ) W )
        y = torch.mm(y, self.emb_e.weight.transpose(1, 0))  # f( vec( f([e_s;rel]) ) W ) e_o
        y += self.b.expand_as(y)

        #y = torch.sigmoid(y)

        return y, (lhs, rel, rhs)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.emb_e.weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    # This is not used in the project but still implemented
    def get_queries(self, queries: torch.Tensor):
        lhs = self.emb_e(queries[:, 0])
        rel = self.emb_rel(queries[:, 1])

        batch_size = len(queries)

        e1_embedded = lhs.view(-1, 1, self.H_dim, self.W_dim)
        rel_embedded = rel.view(-1, 1, self.H_dim, self.W_dim)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)  # [e_s; rel]
        y = self.inp_drop(stacked_inputs)
        y = self.conv1(y)  # [e_s; rel] * w
        y = self.bn1(y)
        y = F.relu(y)  # f([e_s; rel] * w
        y = self.feature_map_drop(y)  # vec( f([e_s;rel]) )
        y = y.view(batch_size, -1)
        y = self.fc(y)  # vec( f([e_s;rel]) ) W
        y = self.hidden_drop(y)
        y = self.bn2(y)
        y = F.relu(y)  # f( vec( f([e_s;rel]) ) W )

        return y


class CP(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(CP, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.lhs = nn.Embedding(sizes[0], rank, sparse=True)
        self.rel = nn.Embedding(sizes[1], rank, sparse=True)
        self.rhs = nn.Embedding(sizes[2], rank, sparse=True)

        self.lhs.weight.data *= init_size
        self.rel.weight.data *= init_size
        self.rhs.weight.data *= init_size

    def score(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        return torch.sum(lhs * rel * rhs, 1, keepdim=True)

    def forward(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])
        return (lhs * rel) @ self.rhs.weight.t(), (lhs, rel, rhs)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        return self.lhs(queries[:, 0]).data * self.rel(queries[:, 1]).data


class ComplEx(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        return (
            (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
        ), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)