
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import torch
from torch import nn
import numpy as np
import random
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
                            scores[i, np.array(filter_in_chunk, dtype=np.int64)] = -1e6

                        else:
                            scores[i, np.array(filter_out, dtype=np.int64)] = -1e6

                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores > targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks

Sigmoid = nn.Sigmoid()

class CP(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(CP, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.context_flag = 0

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
        self.context_flag = 0

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


class ContExt(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int, nb_list:np.ndarray,
            slice_dic: np.ndarray, max_NB: int=50, init_size: float=1e-3,
            data_name: str='FB15K', ascending=1, dropout_1=0.5, dropout_g=0.5,
            evaluation_mode = False
    ):
        super(ContExt, self).__init__()
        n_s, n_r, n_o = sizes
        self.sizes = [n_s, n_r, n_o, n_o]  #append another n_o for nb_o
        self.rank = rank
        self.data_name = data_name
        self.context_flag = 1
        self.flag = 0
        self.ascending = ascending
        self.evaluation_mode = evaluation_mode
        self.padding_idx = n_o

        self.embeddings = nn.ModuleList([
            nn.Embedding(n_s, 2 * rank, sparse=True),
            nn.Embedding(n_r, 2 * rank, sparse=True),
            nn.Embedding(n_o+1, 2 * rank, sparse=True, padding_idx=self.padding_idx)
        ])

        # self.embeddings = nn.ModuleList([
        #     nn.Embedding(s, 2 * rank, sparse=True)
        #     for s in self.sizes[:2]
        # ])

        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size  # For context

        self.W = nn.ParameterList([nn.Parameter(torch.randn((rank*2, rank))), nn.Parameter(torch.randn((rank*2, rank)))])
        self.b_w = nn.ParameterList([nn.Parameter(torch.randn((1, rank))), nn.Parameter(torch.randn((1, rank)))])

        nn.init.xavier_uniform_(self.W[0])
        nn.init.xavier_uniform_(self.W[1])

        nn.init.xavier_uniform_(self.b_w[0])
        nn.init.xavier_uniform_(self.b_w[1])

        self.drop_layer1 = nn.Dropout(p=dropout_1)
        self.drop_layer_g = nn.Dropout(p=dropout_g)

        self.Wo = nn.ParameterList([nn.Parameter(torch.randn((rank, 1))), nn.Parameter(torch.randn((rank, 1)))])
        self.b_g = nn.Parameter(torch.randn((1, 1)))
        self.Uo = nn.ParameterList([nn.Parameter(torch.randn((rank, 1))), nn.Parameter(torch.randn((rank, 1)))])

        nn.init.xavier_uniform_(self.Wo[0])
        nn.init.xavier_uniform_(self.Uo[0])
        nn.init.xavier_uniform_(self.Wo[1])
        nn.init.xavier_uniform_(self.Uo[1])

        nn.init.xavier_uniform_(self.b_g)

        self.nb_list = nb_list
        self.slice_dic = slice_dic

        # self.nb_list = torch.cuda.IntTensor(nb_list)
        # self.slice_dic = torch.cuda.IntTensor(slice_dic)
        self.max_NB = max_NB

    # def get_neighbor(self, subj: torch.Tensor, forward_flag: bool = True, obj: torch.Tensor = None):
    def get_neighbor(self, subj: np.array, forward_flag: bool = True, obj: np.array = None):
        # if forward_flag = False -> obj = None
        index_array = torch.full((len(subj), self.max_NB), self.padding_idx, dtype=torch.long).cuda()

        for i, each_subj in enumerate(subj):
            start_i, end_i = self.slice_dic[each_subj]
            length = end_i - start_i

            if length > 0:
                nb_idx = self.nb_list[start_i:end_i]

                if forward_flag:
                    if np.count_nonzero(nb_idx == obj[i]) <= 1:
                        nb_idx = nb_idx[nb_idx != obj[i]]

                nb_idx = np.random.permutation(np.unique(nb_idx))
                max_len = max([self.max_NB, len(nb_idx)])
                index_array[i, :max_len] = nb_idx

        self.index_array = index_array.clone().data.cpu().numpy()

        return self.embeddings[2](index_array)

    def score(self, x: torch.Tensor):

        self.chunk_size = len(x)

        if self.evaluation_mode:
            self.spo = x

        self.flag += 1

        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        # Concatenation of lhs, rel
        trp_E = torch.cat((lhs[0], rel[0]), dim=1), torch.cat((lhs[1], rel[1]), dim=1)

        w = (trp_E[0] @ self.W[0] - trp_E[1] @ self.W[1] + self.b_w[0],
             trp_E[0] @ self.W[1] + trp_E[1] @ self.W[0] + self.b_w[1])

        nb_E = self.get_neighbor(x[:, 0], forward_flag=False)
        nb_E = nb_E[:, :, :self.rank], nb_E[:, :, self.rank:]  # check on this

        w_nb_E = torch.einsum('bk,bmk->bm', w[0], nb_E[0]) - torch.einsum('bk,bmk->bm', w[1], nb_E[1])
        w_nb_E = torch.where(w_nb_E == 0., torch.tensor(-float('inf')).cuda(), w_nb_E)

        self.alpha = torch.softmax(w_nb_E, dim=1)

        e_c = torch.einsum('bm,bmk->bk', self.alpha, nb_E[0]), torch.einsum('bm,bmk->bk', self.alpha, nb_E[1])

        # calculation of g
        self.g = Sigmoid((lhs[0]*rel[0]-lhs[1]*rel[1]) @ self.Uo[0] - (lhs[1]*rel[0]+lhs[0]*rel[1]) @ self.Uo[1]
                         + e_c[0] @ self.Wo[0] + self.b_g)

        gated_e_c = (self.g * e_c[0] + (torch.ones((self.chunk_size, 1)).cuda() - self.g)*torch.ones_like(e_c[0]),
                     self.g * e_c[1])

        srrr = lhs[0] * rel[0]
        siri = lhs[1] * rel[1]
        sirr = lhs[1] * rel[0]
        srri = lhs[0] * rel[1]

        return torch.sum(((srrr - siri) * gated_e_c[0] + (sirr + srri) * gated_e_c[1]) * rhs[0] +
                         ((sirr + srri) * gated_e_c[0] + (siri - srrr) * gated_e_c[1]) * rhs[1]
                         , 1, keepdim=True)

    def forward(self, x):

        self.chunk_size = len(x)
        self.flag += 1

        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        # Concatenation of lhs, rel
        trp_E = (self.drop_layer1(torch.cat((lhs[0], rel[0]), dim=1)),
                 self.drop_layer1(torch.cat((lhs[1], rel[1]), dim=1)))

        w = (trp_E[0] @ self.W[0] - trp_E[1] @ self.W[1] + self.b_w[0],
             trp_E[0] @ self.W[1] + trp_E[1] @ self.W[0] + self.b_w[1])

        nb_E = self.get_neighbor(x[:, 0], forward_flag=True, obj=x[:, 2])
        nb_E = nb_E[:, :, :self.rank], nb_E[:, :, self.rank:]

        w_nb_E = torch.einsum('bk,bmk->bm', w[0], nb_E[0]) - torch.einsum('bk,bmk->bm', w[1], nb_E[1])
        w_nb_E = torch.where(w_nb_E == 0., torch.tensor(-float('inf')).cuda(), w_nb_E)

        self.alpha = torch.softmax(w_nb_E, dim=1)

        e_c = (torch.einsum('bm,bmk->bk', self.alpha, nb_E[0]),
               torch.einsum('bm,bmk->bk', self.alpha, nb_E[1]))

        # calculation of g
        self.g = Sigmoid((lhs[0]*rel[0]-lhs[1]*rel[1])@ self.Uo[0] - (lhs[1]*rel[0]+lhs[0]*rel[1])@ self.Uo[1]
                         + e_c[0] @ self.Wo[0] + self.b_g)
        g = self.drop_layer_g(self.g)

        gated_e_c = (g * e_c[0] + (torch.ones((self.chunk_size, 1)).cuda() - g) * torch.ones_like(e_c[0]), g * e_c[1])

        srrr = lhs[0] * rel[0]
        siri = lhs[1] * rel[1]
        sirr = lhs[1] * rel[0]
        srri = lhs[0] * rel[1]

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]

        return (
               ((srrr - siri) * gated_e_c[0] + (sirr + srri) * gated_e_c[1]) @ to_score[0].transpose(0, 1) +
               ((srri + sirr) * gated_e_c[0] + (siri - srrr) * gated_e_c[1]) @ to_score[1].transpose(0, 1)
        ), (
           torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
           torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
           torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2),
           g * torch.sqrt(e_c[0]**2 + e_c[1] ** 2)
        )

    def get_queries(self, queries: torch.Tensor):

        self.chunk_size = len(queries)
        self.flag += 1

        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        # Concatenation of lhs, rel
        trp_E = torch.cat((lhs[0], rel[0]), dim=1), torch.cat((lhs[1], rel[1]), dim=1)

        w = (trp_E[0] @ self.W[0] - trp_E[1] @ self.W[1] + self.b_w[0],
             trp_E[0] @ self.W[1] + trp_E[1] @ self.W[0] + self.b_w[1])

        nb_E = self.get_neighbor(queries[:, 0], forward_flag=False)
        nb_E = nb_E[:, :, :self.rank], nb_E[:, :, self.rank:]  # check on this

        w_nb_E = torch.einsum('bk,bmk->bm', w[0], nb_E[0]) - torch.einsum('bk,bmk->bm', w[1], nb_E[1])
        w_nb_E = torch.where(w_nb_E == 0., torch.tensor(-float('inf')).cuda(), w_nb_E)

        self.alpha = torch.softmax(w_nb_E, dim=1)

        e_c = torch.einsum('bm,bmk->bk', self.alpha, nb_E[0]), torch.einsum('bm,bmk->bk', self.alpha, nb_E[1])

        # calculation of g
        self.g = Sigmoid((lhs[0] * rel[0] - lhs[1] * rel[1]) @ self.Uo[0]
                         - (lhs[1] * rel[0] + lhs[0] * rel[1]) @ self.Uo[1]
                         + e_c[0] @ self.Wo[0] + self.b_g)

        gated_e_c = (self.g * e_c[0] + (torch.ones((self.chunk_size, 1)).cuda() - self.g) * torch.ones_like(e_c[0]),
                     self.g * e_c[1])

        srrr = lhs[0] * rel[0]
        siri = lhs[1] * rel[1]
        sirr = lhs[1] * rel[0]
        srri = lhs[0] * rel[1]

        return torch.cat(((srrr - siri) * gated_e_c[0] + (sirr + srri) * gated_e_c[1],
                          (srri + sirr) * gated_e_c[0] + (siri - srrr) * gated_e_c[1]), 1)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)