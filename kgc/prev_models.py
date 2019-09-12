
# Using s, r, o, e_c embedding for context CP

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

'''
Correction -> requires_grad = True
'''

class Context_CP(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int, sorted_data: np.ndarray,
            slice_dic: np.ndarray, max_NB: int = 50, init_size: float = 1e-3,
            data_name: str = 'FB15K', ascending=1
    ):
        super(Context_CP, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.data_name = data_name
        self.context_flag = 1
        self.ascending = ascending
        self.flag = 0

        self.lhs = nn.Embedding(sizes[0], rank, sparse=True)
        self.rel = nn.Embedding(sizes[1], rank, sparse=True)
        self.rhs = nn.Embedding(sizes[2], rank, sparse=True)
        self.ctxt = nn.Embedding(sizes[2], rank, sparse=True)
        # Embedding for context

        self.lhs.weight.data *= init_size
        self.rel.weight.data *= init_size
        self.rhs.weight.data *= init_size
        self.ctxt.weight.data *= init_size

        # Context related parameters
        self.W = nn.Linear(int(2 * rank), rank, bias=True)  # W for w = [lhs; rel; rhs]^T W
        # self.bn1 = nn.BatchNorm1d(rank).cuda()
        self.W2 = nn.Linear(rank, rank, bias=True)
        # self.bn2 = nn.BatchNorm1d(rank).cuda()

        self.drop_layer1 = nn.Dropout(p=0.5)  # apply dropout to only forward
        self.drop_layer2 = nn.Dropout(p=0.5)

        # Weights for the gate (added)
        self.Wo = nn.Linear(rank, 1, bias=True)
        self.Uo = nn.Linear(rank, 1, bias=True)

        nn.init.xavier_uniform_(self.W.weight)  # Xavier initialization
        nn.init.xavier_uniform_(self.W2.weight)

        self.sorted_data = sorted_data
        self.slice_dic = slice_dic
        self.max_NB = max_NB

    def get_neighbor(self, subj: torch.Tensor):
        # return neighbor (N_subject, N_nb_max, k)
        index_array = np.zeros(shape=(len(subj), self.max_NB), dtype=np.int32)

        for i, each_subj in enumerate(subj):
            _, start_i, end_i = self.slice_dic[each_subj]
            length = end_i - start_i

            if length > 0:
                if self.max_NB >= length:
                    index_array[i, :length] = self.sorted_data[start_i:end_i, 2]
                else:  # Need to uniformly truncate
                    hop = int(length / self.max_NB)
                    index_array[i, :] = self.sorted_data[start_i:end_i:hop, 2][:self.max_NB]
                if self.ascending == -1:
                    index_array[i, :] = index_array[i, :][::-1]

        # Convert index_array into a long tensor for indexing the embedding.
        index_tensor = torch.LongTensor(index_array).cuda()

        return self.ctxt(index_tensor)

    def score(self, x: torch.Tensor):

        self.chunk_size = len(x)
        self.flag += 1

        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        # concatenation of lhs, rel
        trp_E = torch.cat((lhs, rel), dim=1)  # (previous)

        # Get attention weight vector, where W.shape == (3k, k)
        # w = self.bn1(self.W(trp_E))  # w.shape == (chunk_size, k) and batch-norm
        w = self.W(trp_E)  # w.shape == (chunk_size, k) and batch-norm
        # Get nb_E
        nb_E = self.get_neighbor(x[:, 0])  # nb_E.shape == (chunk_size, max_NB, k)

        self.alpha = torch.softmax(torch.einsum('bk,bmk->bm', w, nb_E), dim=1)
        # alpha.shape == (chunk_size, max_NB)

        # Get context vector
        e_c = self.W2(torch.einsum('bm,bmk->bk', self.alpha, nb_E))
        # extra linear layer and batch-norm

        # Gate
        self.g = Sigmoid(self.Uo(lhs*rel) + self.Wo(e_c))

        gated_e_c = self.g * e_c + (torch.ones((self.chunk_size, 1)).cuda() - self.g) * torch.ones_like(e_c).cuda()

        # Get tot_score
        tot_score = torch.sum(lhs * rel * rhs * gated_e_c, 1, keepdim=True)

        return tot_score

    def forward(self, x: torch.Tensor):

        self.chunk_size = len(x)
        self.flag += 1

        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        # concatenation of lhs, rel, rhs
        trp_E = torch.cat((lhs, rel), dim=1)  # (previous)

        # Get attention weight vector, where W.shape == (3k, k)
        w = self.W(self.drop_layer1(trp_E))
        # w.shape == (chunk_size, k) and batch-norm, dropout

        # Get nb_E
        nb_E = self.get_neighbor(x[:, 0])  # nb_E.shape == (chunk_size, max_NB, k)

        self.alpha = torch.softmax(torch.einsum('bk,bmk->bm', w, nb_E), dim=1)
        # alpha.shape == (chunk_size, max_NB)

        e_c = self.W2(self.drop_layer2(torch.einsum('bm,bmk->bk', self.alpha, nb_E)))
        # extra linear layer and batch-normalization

        # Gate
        self.g = Sigmoid(self.Uo(lhs * rel) + self.Wo(e_c))

        gated_e_c = self.g * e_c + (torch.ones((self.chunk_size, 1)).cuda() - self.g) * torch.ones_like(e_c).cuda()

        # Get tot_score
        tot_forward = (lhs * rel * gated_e_c) @ self.rhs.weight.t()

        return tot_forward, (lhs, rel, rhs, gated_e_c)

    def get_queries(self, x: torch.Tensor):  # need to include context part
        # x is a numpy array (equivalent to queries)

        self.chunk_size = len(x)
        self.flag += 1

        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])

        # concatenation of lhs, rel, rhs
        trp_E = torch.cat((lhs, rel), dim=1)  # trp_E.shape == (chunk_size, 3k) previous

        # w = self.bn1(self.W(trp_E))  # w.shape == (chunk_size, k) and added batch-norm
        w = self.W(trp_E)  # w.shape == (chunk_size, k) and added batch-norm
        nb_E = self.get_neighbor(x[:, 0])  # nb_E.shape == (chunk_size, max_NB, k)

        self.alpha = torch.softmax(torch.einsum('bk,bmk->bm', w, nb_E), dim=1)
        # alpha.shape == (chunk_size, max_NB)

        e_c = self.W2(torch.einsum('bm,bmk->bk', self.alpha, nb_E))
        self.g = Sigmoid(self.Uo(lhs * rel) + self.Wo(e_c))

        gated_e_c = self.g * e_c + (torch.ones((self.chunk_size, 1)).cuda() - self.g) * torch.ones_like(e_c).cuda()

        return lhs.data * rel.data * gated_e_c.data

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

'''
Correction -> nn.Parameter(tensor)
'''
# Fix the requires_grad=True problem


class Context_ComplEx(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int, sorted_data:np.ndarray,
            slice_dic: np.ndarray, max_NB: int=50, init_size: float=1e-3,
            data_name: str='FB15K', ascending=1
    ):
        super(Context_ComplEx, self).__init__()
        n_s, n_r, n_o = sizes
        self.sizes = [n_s, n_r, n_o, n_o]  #append another n_o for nb_o
        self.rank = rank
        self.data_name = data_name
        self.context_flag = 1
        self.flag = 0
        self.ascending = ascending

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in self.sizes[:3]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size  # For context

        self.W = nn.ParameterList([nn.Parameter(torch.randn((rank*2, rank))), nn.Parameter(torch.randn((rank*2, rank)))])
        self.b_w = nn.ParameterList([nn.Parameter(torch.randn((1, rank))), nn.Parameter(torch.randn((1, rank)))])
        self.W2 = nn.ParameterList([nn.Parameter(torch.randn((rank, rank))), nn.Parameter(torch.randn((rank, rank)))])
        self.b_w2 = nn.ParameterList([nn.Parameter(torch.randn((1, rank))), nn.Parameter(torch.randn((1, rank)))])

        nn.init.xavier_uniform_(self.W[0])
        nn.init.xavier_uniform_(self.W2[0])
        nn.init.xavier_uniform_(self.W[1])
        nn.init.xavier_uniform_(self.W2[1])

        nn.init.xavier_uniform_(self.b_w[0])
        nn.init.xavier_uniform_(self.b_w[1])
        nn.init.xavier_uniform_(self.b_w2[0])
        nn.init.xavier_uniform_(self.b_w2[1])

        self.drop_layer1 = nn.Dropout(p=0.5)
        self.drop_layer2 = nn.Dropout(p=0.5)

        self.Wo = nn.ParameterList([nn.Parameter(torch.randn((rank, 1))), nn.Parameter(torch.randn((rank, 1)))])
        self.b_g = nn.Parameter(torch.randn((1, 1)))
        self.Uo = nn.ParameterList([nn.Parameter(torch.randn((rank, 1))), nn.Parameter(torch.randn((rank, 1)))])

        nn.init.xavier_uniform_(self.Wo[0])
        nn.init.xavier_uniform_(self.Uo[0])
        nn.init.xavier_uniform_(self.Wo[1])
        nn.init.xavier_uniform_(self.Uo[1])

        nn.init.xavier_uniform_(self.b_g)

        self.sorted_data = sorted_data
        self.slice_dic = slice_dic
        self.max_NB = max_NB

    def get_neighbor(self, subj: torch.Tensor):
        index_array = np.zeros(shape=(len(subj), self.max_NB), dtype=np.int32)

        for i, each_subj in enumerate(subj):
            _, start_i, end_i = self.slice_dic[each_subj]
            length = end_i - start_i

            if length > 0:
                if self.max_NB >= length:
                    index_array[i, :length] = self.sorted_data[start_i:end_i, 2]
                else:  # Need to uniformly truncate
                    hop = int(length / self.max_NB)
                    index_array[i, :] = self.sorted_data[start_i:end_i:hop, 2][:self.max_NB]
                if self.ascending == -1:
                    index_array[i, :] = index_array[i, :][::-1]

        index_tensor = torch.LongTensor(index_array).cuda()

        return self.embeddings[2](index_tensor)

    def score(self, x: torch.Tensor):

        self.chunk_size = len(x)
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

        nb_E = self.get_neighbor(x[:, 0])
        nb_E = nb_E[:, :, :self.rank], nb_E[:, :, self.rank:]  # check on this

        # Take the real part of w @ nb_E
        self.alpha = torch.softmax(torch.einsum('bk,bmk->bm', w[0], nb_E[0]) - torch.einsum('bk,bmk->bm', w[1], nb_E[1]),
                                   dim=1)

        e_c = torch.einsum('bm,bmk->bk', self.alpha, nb_E[0]), torch.einsum('bm,bmk->bk', self.alpha, nb_E[1])

        # Linear matrix multiplication
        e_c = (e_c[0] @ self.W2[0] - e_c[1] @ self.W2[1] + self.b_w2[0],
               e_c[0] @ self.W2[1] + e_c[1] @ self.W2[0] + self.b_w2[1])

        # calculation of g
        self.g = Sigmoid((lhs[0]*rel[0]-lhs[1]*rel[1]) @ self.Uo[0] - (lhs[1]*rel[0]+lhs[0]*rel[1]) @ self.Uo[1]
                          + e_c[0] @ self.Wo[0] + self.b_g)

        gated_e_c = (self.g * e_c[0] + (torch.ones((self.chunk_size, 1)).cuda() - self.g)*torch.ones_like(e_c[0]),
                     self.g * e_c[1])

        srrr = lhs[0] * rel[0]
        siri = lhs[1] * rel[1]
        sirr = lhs[1] * rel[0]
        srri = lhs[0] * rel[1]

        return torch.sum(((srrr - siri) * gated_e_c[0] + (sirr + srri) * gated_e_c[1])*rhs[0] +
                         ((sirr + srri) * gated_e_c[0] + (siri - srrr) * gated_e_c[1])*rhs[1]
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

        nb_E = self.get_neighbor(x[:, 0])
        nb_E = nb_E[:, :, :self.rank], nb_E[:, :, self.rank:]  # check on this

        # Take the real part of w @ nb_E
        self.alpha = torch.softmax(torch.einsum('bk,bmk->bm', w[0], nb_E[0]) - torch.einsum('bk,bmk->bm', w[1], nb_E[1]),
                                   dim=1)

        e_c = (self.drop_layer2(torch.einsum('bm,bmk->bk', self.alpha, nb_E[0])),
               self.drop_layer2(torch.einsum('bm,bmk->bk', self.alpha, nb_E[1])))

        # Need dropout applied to e_c

        # Linear matrix multiplication
        e_c = (e_c[0] @ self.W2[0] - e_c[1] @ self.W2[1] + self.b_w2[0],
               e_c[0] @ self.W2[1] + e_c[1] @ self.W2[0] + self.b_w2[1])

        # calculation of g
        self.g = Sigmoid((lhs[0]*rel[0]-lhs[1]*rel[1])@ self.Uo[0] - (lhs[1]*rel[0]+lhs[0]*rel[1])@ self.Uo[1]
                    + e_c[0] @ self.Wo[0] + self.b_g)

        gated_e_c = (self.g * e_c[0] + (torch.ones((self.chunk_size, 1)).cuda() - self.g) * torch.ones_like(e_c[0]),
                     self.g * e_c[1])

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
            torch.sqrt(lhs[0]**2 + lhs[1]**2),
            torch.sqrt(rel[0]**2 + rel[1]**2),
            torch.sqrt(rhs[0]**2 + rhs[1]**2),
            torch.sqrt(gated_e_c[0]**2 + gated_e_c[1]**2)
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

        nb_E = self.get_neighbor(queries[:, 0])
        nb_E = nb_E[:, :, :self.rank], nb_E[:, :, self.rank:]  # check on this

        # Take the real part of w @ nb_E
        self.alpha = torch.softmax(torch.einsum('bk,bmk->bm', w[0], nb_E[0]) - torch.einsum('bk,bmk->bm', w[1], nb_E[1]),
                                   dim=1)

        e_c = torch.einsum('bm,bmk->bk', self.alpha, nb_E[0]), torch.einsum('bm,bmk->bk', self.alpha, nb_E[1])

        # Linear matrix multiplication
        e_c = (e_c[0] @ self.W2[0] - e_c[1] @ self.W2[1] + self.b_w2[0],
               e_c[0] @ self.W2[1] + e_c[1] @ self.W2[0] + self.b_w2[1])

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

'''
v2:
Removes the linear layer which is used to calculate the neighborhood-context vector
'''
class Context_CP_v2(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int, sorted_data: np.ndarray,
            slice_dic: np.ndarray, max_NB: int = 50, init_size: float = 1e-3,
            data_name: str = 'FB15K', ascending=1
    ):
        super(Context_CP_v2, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.data_name = data_name
        self.context_flag = 1
        self.ascending = ascending
        self.flag = 0

        self.lhs = nn.Embedding(sizes[0], rank, sparse=True)
        self.rel = nn.Embedding(sizes[1], rank, sparse=True)
        self.rhs = nn.Embedding(sizes[2], rank, sparse=True)
        self.ctxt = nn.Embedding(sizes[2], rank, sparse=True)
        # Embedding for context

        self.lhs.weight.data *= init_size
        self.rel.weight.data *= init_size
        self.rhs.weight.data *= init_size
        self.ctxt.weight.data *= init_size

        # Context related parameters
        self.W = nn.Linear(int(2 * rank), rank, bias=True)  # W for w = [lhs; rel; rhs]^T W

        self.drop_layer1 = nn.Dropout(p=0.5)  # apply dropout to only forward

        # Weights for the gate (added)
        self.Wo = nn.Linear(rank, 1, bias=True)
        self.Uo = nn.Linear(rank, 1, bias=True)

        nn.init.xavier_uniform_(self.W.weight)  # Xavier initialization

        self.sorted_data = sorted_data
        self.slice_dic = slice_dic
        self.max_NB = max_NB

    def get_neighbor(self, subj: torch.Tensor):
        # return neighbor (N_subject, N_nb_max, k)
        index_array = np.zeros(shape=(len(subj), self.max_NB), dtype=np.int32)

        for i, each_subj in enumerate(subj):
            _, start_i, end_i = self.slice_dic[each_subj]
            length = end_i - start_i

            if length > 0:
                if self.max_NB >= length:
                    index_array[i, :length] = self.sorted_data[start_i:end_i, 2]
                else:  # Need to uniformly truncate
                    hop = int(length / self.max_NB)
                    index_array[i, :] = self.sorted_data[start_i:end_i:hop, 2][:self.max_NB]
                if self.ascending == -1:
                    index_array[i, :] = index_array[i, :][::-1]

        # Convert index_array into a long tensor for indexing the embedding.
        index_tensor = torch.LongTensor(index_array).cuda()

        return self.ctxt(index_tensor)

    def score(self, x: torch.Tensor):

        self.chunk_size = len(x)
        self.flag += 1

        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        # concatenation of lhs, rel
        trp_E = torch.cat((lhs, rel), dim=1)  # (previous)

        # Get attention weight vector, where W.shape == (3k, k)
        # w = self.bn1(self.W(trp_E))  # w.shape == (chunk_size, k) and batch-norm
        w = self.W(trp_E)  # w.shape == (chunk_size, k) and batch-norm
        # Get nb_E
        nb_E = self.get_neighbor(x[:, 0])  # nb_E.shape == (chunk_size, max_NB, k)

        self.alpha = torch.softmax(torch.einsum('bk,bmk->bm', w, nb_E), dim=1)
        # alpha.shape == (chunk_size, max_NB)

        # Get context vector
        e_c = torch.einsum('bm,bmk->bk', self.alpha, nb_E)
        # extra linear layer and batch-norm

        # Gate
        self.g = Sigmoid(self.Uo(lhs*rel) + self.Wo(e_c))

        gated_e_c = self.g * e_c + (torch.ones((self.chunk_size, 1)).cuda() - self.g) * torch.ones_like(e_c).cuda()

        # Get tot_score
        tot_score = torch.sum(lhs * rel * rhs * gated_e_c, 1, keepdim=True)

        return tot_score

    def forward(self, x: torch.Tensor):

        self.chunk_size = len(x)
        self.flag += 1

        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        # concatenation of lhs, rel, rhs
        trp_E = torch.cat((lhs, rel), dim=1)  # (previous)

        # Get attention weight vector, where W.shape == (3k, k)
        w = self.W(self.drop_layer1(trp_E))
        # w.shape == (chunk_size, k) and batch-norm, dropout

        # Get nb_E
        nb_E = self.get_neighbor(x[:, 0])  # nb_E.shape == (chunk_size, max_NB, k)

        self.alpha = torch.softmax(torch.einsum('bk,bmk->bm', w, nb_E), dim=1)
        # alpha.shape == (chunk_size, max_NB)

        e_c = torch.einsum('bm,bmk->bk', self.alpha, nb_E)
        # extra linear layer and batch-normalization

        # Gate
        self.g = Sigmoid(self.Uo(lhs * rel) + self.Wo(e_c))

        gated_e_c = self.g * e_c + (torch.ones((self.chunk_size, 1)).cuda() - self.g) * torch.ones_like(e_c).cuda()

        # Get tot_score
        tot_forward = (lhs * rel * gated_e_c) @ self.rhs.weight.t()

        return tot_forward, (lhs, rel, rhs, gated_e_c)

    def get_queries(self, x: torch.Tensor):  # need to include context part
        # x is a numpy array (equivalent to queries)

        self.chunk_size = len(x)
        self.flag += 1

        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])

        # concatenation of lhs, rel, rhs
        trp_E = torch.cat((lhs, rel), dim=1)  # trp_E.shape == (chunk_size, 3k) previous

        # w = self.bn1(self.W(trp_E))  # w.shape == (chunk_size, k) and added batch-norm
        w = self.W(trp_E)  # w.shape == (chunk_size, k) and added batch-norm
        nb_E = self.get_neighbor(x[:, 0])  # nb_E.shape == (chunk_size, max_NB, k)

        self.alpha = torch.softmax(torch.einsum('bk,bmk->bm', w, nb_E), dim=1)
        # alpha.shape == (chunk_size, max_NB)

        e_c = torch.einsum('bm,bmk->bk', self.alpha, nb_E)
        self.g = Sigmoid(self.Uo(lhs * rel) + self.Wo(e_c))

        gated_e_c = self.g * e_c + (torch.ones((self.chunk_size, 1)).cuda() - self.g) * torch.ones_like(e_c).cuda()

        return lhs.data * rel.data * gated_e_c.data

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)


'''
Fix g = 0
'''
class Context_ComplEx_v2(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int, sorted_data:np.ndarray,
            slice_dic: np.ndarray, max_NB: int=50, init_size: float=1e-3,
            data_name: str='FB15K', ascending=1
    ):
        super(Context_ComplEx_v2, self).__init__()
        n_s, n_r, n_o = sizes
        self.sizes = [n_s, n_r, n_o, n_o]  #append another n_o for nb_o
        self.rank = rank
        self.data_name = data_name
        self.context_flag = 1
        self.flag = 0
        self.ascending = ascending

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in self.sizes[:3]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size  # For context

        self.W = nn.ParameterList([nn.Parameter(torch.randn((rank*2, rank))), nn.Parameter(torch.randn((rank*2, rank)))])
        self.b_w = nn.ParameterList([nn.Parameter(torch.randn((1, rank))), nn.Parameter(torch.randn((1, rank)))])

        nn.init.xavier_uniform_(self.W[0])
        nn.init.xavier_uniform_(self.W[1])

        nn.init.xavier_uniform_(self.b_w[0])
        nn.init.xavier_uniform_(self.b_w[1])

        self.drop_layer1 = nn.Dropout(p=0.5)

        self.Wo = nn.ParameterList([nn.Parameter(torch.randn((rank, 1))), nn.Parameter(torch.randn((rank, 1)))])
        self.b_g = nn.Parameter(torch.randn((1, 1)))
        self.Uo = nn.ParameterList([nn.Parameter(torch.randn((rank, 1))), nn.Parameter(torch.randn((rank, 1)))])

        nn.init.xavier_uniform_(self.Wo[0])
        nn.init.xavier_uniform_(self.Uo[0])
        nn.init.xavier_uniform_(self.Wo[1])
        nn.init.xavier_uniform_(self.Uo[1])

        nn.init.xavier_uniform_(self.b_g)

        self.sorted_data = sorted_data
        self.slice_dic = slice_dic
        self.max_NB = max_NB

    def get_neighbor(self, subj: torch.Tensor):
        index_array = np.zeros(shape=(len(subj), self.max_NB), dtype=np.int32)

        for i, each_subj in enumerate(subj):
            _, start_i, end_i = self.slice_dic[each_subj]
            length = end_i - start_i

            if length > 0:
                if self.max_NB >= length:
                    index_array[i, :length] = self.sorted_data[start_i:end_i, 2]
                else:  # Need to uniformly truncate
                    hop = int(length / self.max_NB)
                    index_array[i, :] = self.sorted_data[start_i:end_i:hop, 2][:self.max_NB]
                if self.ascending == -1:
                    index_array[i, :] = index_array[i, :][::-1]

        index_tensor = torch.LongTensor(index_array).cuda()

        return self.embeddings[2](index_tensor)

    def score(self, x: torch.Tensor):

        self.chunk_size = len(x)
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

        nb_E = self.get_neighbor(x[:, 0])
        nb_E = nb_E[:, :, :self.rank], nb_E[:, :, self.rank:]  # check on this

        # Take the real part of w @ nb_E
        self.alpha = torch.softmax(torch.einsum('bk,bmk->bm', w[0], nb_E[0]) - torch.einsum('bk,bmk->bm', w[1], nb_E[1]),
                                   dim=1)

        e_c = torch.einsum('bm,bmk->bk', self.alpha, nb_E[0]), torch.einsum('bm,bmk->bk', self.alpha, nb_E[1])

        # calculation of g
        self.g = Sigmoid((lhs[0]*rel[0]-lhs[1]*rel[1]) @ self.Uo[0] - (lhs[1]*rel[0]+lhs[0]*rel[1]) @ self.Uo[1]
                         + e_c[0] @ self.Wo[0] + self.b_g)

        self.g = torch.zeros((self.chunk_size, 1)).cuda()

        gated_e_c = (self.g * e_c[0] + (torch.ones((self.chunk_size, 1)).cuda() - self.g)*torch.ones_like(e_c[0]).cuda(),
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

        nb_E = self.get_neighbor(x[:, 0])
        nb_E = nb_E[:, :, :self.rank], nb_E[:, :, self.rank:]  # check on this

        # Take the real part of w @ nb_E
        self.alpha = torch.softmax(torch.einsum('bk,bmk->bm', w[0], nb_E[0]) - torch.einsum('bk,bmk->bm', w[1], nb_E[1]),
                                   dim=1)

        e_c = (torch.einsum('bm,bmk->bk', self.alpha, nb_E[0]),
               torch.einsum('bm,bmk->bk', self.alpha, nb_E[1]))

        # calculation of g
        self.g = Sigmoid((lhs[0]*rel[0]-lhs[1]*rel[1])@ self.Uo[0] - (lhs[1]*rel[0]+lhs[0]*rel[1])@ self.Uo[1]
                         + e_c[0] @ self.Wo[0] + self.b_g)

        self.g = torch.zeros((self.chunk_size, 1)).cuda()

        gated_e_c = (self.g * e_c[0] + (torch.ones((self.chunk_size, 1)).cuda() - self.g) * torch.ones_like(e_c[0]),
                     self.g * e_c[1])

        srrr = lhs[0] * rel[0]
        siri = lhs[1] * rel[1]
        sirr = lhs[1] * rel[0]
        srri = lhs[0] * rel[1]

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]

        return (
                ((srrr - siri) * gated_e_c[0] + (srri - sirr) * gated_e_c[1]) @ to_score[0].transpose(0, 1) +
                ((srri + sirr) * gated_e_c[0] + (siri - srrr) * gated_e_c[1]) @ to_score[1].transpose(0, 1)
        ), (
            torch.sqrt(lhs[0]**2 + lhs[1]**2),
            torch.sqrt(rel[0]**2 + rel[1]**2),
            torch.sqrt(rhs[0]**2 + rhs[1]**2),
            torch.sqrt(gated_e_c[0]**2 + gated_e_c[1]**2)
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

        nb_E = self.get_neighbor(queries[:, 0])
        nb_E = nb_E[:, :, :self.rank], nb_E[:, :, self.rank:]  # check on this

        # Take the real part of w @ nb_E
        self.alpha = torch.softmax(torch.einsum('bk,bmk->bm', w[0], nb_E[0]) - torch.einsum('bk,bmk->bm', w[1], nb_E[1]),
                                   dim=1)

        e_c = torch.einsum('bm,bmk->bk', self.alpha, nb_E[0]), torch.einsum('bm,bmk->bk', self.alpha, nb_E[1])

        # calculation of g
        self.g = Sigmoid((lhs[0] * rel[0] - lhs[1] * rel[1]) @ self.Uo[0]
                    - (lhs[1] * rel[0] + lhs[0] * rel[1]) @ self.Uo[1]
                    + e_c[0] @ self.Wo[0] + self.b_g)

        self.g = torch.zeros((self.chunk_size, 1)).cuda()

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


'''
v3:
Correction -> requires_grad = True
Removes the linear layer which is used to calculate the neighborhood-context vector
Adds dropout layer on g
Secondary Pre-training where ComplEx Embedding is Frozen
Use the same embedding as the ComplEx
'''


class Context_CP_v3(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int, sorted_data: np.ndarray,
            slice_dic: np.ndarray, max_NB: int = 50, init_size: float = 1e-3,
            data_name: str = 'FB15K', ascending=1
    ):
        super(Context_CP_v3, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.data_name = data_name
        self.context_flag = 1
        self.ascending = ascending
        self.flag = 0

        self.lhs = nn.Embedding(sizes[0], rank, sparse=True)
        self.rel = nn.Embedding(sizes[1], rank, sparse=True)
        self.rhs = nn.Embedding(sizes[2], rank, sparse=True)
        self.ctxt = nn.Embedding(sizes[2], rank, sparse=True)
        # Embedding for context

        self.lhs.weight.data *= init_size
        self.rel.weight.data *= init_size
        self.rhs.weight.data *= init_size
        self.ctxt.weight.data *= init_size

        # Context related parameters
        self.W = nn.Linear(int(2 * rank), rank, bias=True)  # W for w = [lhs; rel; rhs]^T W

        self.drop_layer1 = nn.Dropout(p=0.5)  # apply dropout to only forward
        self.drop_layer_g = nn.Dropout(p=0.3)

        self.Wo = nn.Linear(rank, 1, bias=True)
        self.Uo = nn.Linear(rank, 1, bias=True)

        nn.init.xavier_uniform_(self.W.weight)  # Xavier initialization
        nn.init.xavier_uniform_(self.W2.weight)

        self.sorted_data = sorted_data
        self.slice_dic = slice_dic
        self.max_NB = max_NB

    def get_neighbor(self, subj: torch.Tensor):
        # return neighbor (N_subject, N_nb_max, k)
        index_array = np.zeros(shape=(len(subj), self.max_NB), dtype=np.int32)

        for i, each_subj in enumerate(subj):
            _, start_i, end_i = self.slice_dic[each_subj]
            length = end_i - start_i

            if length > 0:
                if self.max_NB >= length:
                    index_array[i, :length] = self.sorted_data[start_i:end_i, 2]
                else:  # Need to uniformly truncate
                    hop = int(length / self.max_NB)
                    index_array[i, :] = self.sorted_data[start_i:end_i:hop, 2][:self.max_NB]
                if self.ascending == -1:
                    index_array[i, :] = index_array[i, :][::-1]

        # Convert index_array into a long tensor for indexing the embedding.
        index_tensor = torch.LongTensor(index_array).cuda()

        return self.ctxt(index_tensor)

    def score(self, x: torch.Tensor):

        self.chunk_size = len(x)
        self.flag += 1

        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        # concatenation of lhs, rel
        trp_E = torch.cat((lhs, rel), dim=1)  # (previous)

        # Get attention weight vector, where W.shape == (3k, k)
        # w = self.bn1(self.W(trp_E))  # w.shape == (chunk_size, k) and batch-norm
        w = self.W(trp_E)  # w.shape == (chunk_size, k) and batch-norm
        # Get nb_E
        nb_E = self.get_neighbor(x[:, 0])  # nb_E.shape == (chunk_size, max_NB, k)

        self.alpha = torch.softmax(torch.einsum('bk,bmk->bm', w, nb_E), dim=1)
        # alpha.shape == (chunk_size, max_NB)

        # Get context vector
        e_c = torch.einsum('bm,bmk->bk', self.alpha, nb_E)
        # extra linear layer and batch-norm

        # Gate
        self.g = Sigmoid(self.Uo(lhs*rel) + self.Wo(e_c))

        gated_e_c = self.g * e_c + (torch.ones((self.chunk_size, 1)).cuda() - self.g) * torch.ones_like(e_c).cuda()

        # Get tot_score
        tot_score = torch.sum(lhs * rel * rhs * gated_e_c, 1, keepdim=True)

        return tot_score

    def forward(self, x: torch.Tensor):

        self.chunk_size = len(x)
        self.flag += 1

        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        # concatenation of lhs, rel, rhs
        trp_E = torch.cat((lhs, rel), dim=1)  # (previous)

        # Get attention weight vector, where W.shape == (3k, k)
        w = self.W(self.drop_layer1(trp_E))
        # w.shape == (chunk_size, k) and batch-norm, dropout

        # Get nb_E
        nb_E = self.get_neighbor(x[:, 0])  # nb_E.shape == (chunk_size, max_NB, k)

        self.alpha = torch.softmax(torch.einsum('bk,bmk->bm', w, nb_E), dim=1)
        # alpha.shape == (chunk_size, max_NB)

        e_c = torch.einsum('bm,bmk->bk', self.alpha, nb_E)
        # extra linear layer and batch-normalization

        # Gate
        self.g = Sigmoid(self.Uo(lhs * rel) + self.Wo(e_c))
        self.g = self.drop_layer_g(self.g)

        gated_e_c = self.g * e_c + (torch.ones((self.chunk_size, 1)).cuda() - self.g) * torch.ones_like(e_c).cuda()

        # Get tot_score
        tot_forward = (lhs * rel * gated_e_c) @ self.rhs.weight.t()

        return tot_forward, (lhs, rel, rhs, gated_e_c)

    def get_queries(self, x: torch.Tensor):  # need to include context part
        # x is a numpy array (equivalent to queries)

        self.chunk_size = len(x)
        self.flag += 1

        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])

        # concatenation of lhs, rel, rhs
        trp_E = torch.cat((lhs, rel), dim=1)  # trp_E.shape == (chunk_size, 3k) previous

        # w = self.bn1(self.W(trp_E))  # w.shape == (chunk_size, k) and added batch-norm
        w = self.W(trp_E)  # w.shape == (chunk_size, k) and added batch-norm
        nb_E = self.get_neighbor(x[:, 0])  # nb_E.shape == (chunk_size, max_NB, k)

        self.alpha = torch.softmax(torch.einsum('bk,bmk->bm', w, nb_E), dim=1)
        # alpha.shape == (chunk_size, max_NB)

        e_c = torch.einsum('bm,bmk->bk', self.alpha, nb_E)
        self.g = Sigmoid(self.Uo(lhs * rel) + self.Wo(e_c))

        gated_e_c = self.g * e_c + (torch.ones((self.chunk_size, 1)).cuda() - self.g) * torch.ones_like(e_c).cuda()

        return lhs.data * rel.data * gated_e_c.data

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)


class Context_ComplEx_v3(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int, sorted_data:np.ndarray,
            slice_dic: np.ndarray, max_NB: int=50, init_size: float=1e-3,
            data_name: str='FB15K', ascending=1, dropout_1=0.5, dropout_g=0.5,
            evaluation_mode = False
    ):
        super(Context_ComplEx_v3, self).__init__()
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
        ])  # the index with s+1 -> padding with 0

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

        self.sorted_data = torch.cuda.IntTensor(sorted_data)
        self.slice_dic = torch.cuda.IntTensor(slice_dic)
        self.max_NB = max_NB

    def get_neighbor(self, subj: torch.Tensor):
        index_array = torch.full((len(subj), self.max_NB), self.padding_idx, dtype=torch.long).cuda()

        for i, each_subj in enumerate(subj):
            _, start_i, end_i = self.slice_dic[each_subj]
            length = end_i - start_i

            if length > 0:
                if self.max_NB >= length:  # padded with -1 at the end
                    index_array[i, :length] = self.sorted_data[start_i:end_i, 2][torch.randperm(length).cuda()]
                else:  # Need to uniformly truncate
                    index_array[i, :] = self.sorted_data[start_i:end_i, 2][torch.randperm(length).cuda()][:self.max_NB]

        self.index_array = index_array.clone().data.cpu().numpy()

        return self.embeddings[2](index_array)  #padded embeddings = [0., 0. ,... 0.]

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

        nb_E = self.get_neighbor(x[:, 0])
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

        nb_E = self.get_neighbor(x[:, 0])
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
           torch.sqrt(gated_e_c[0] ** 2 + gated_e_c[1] ** 2)
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

        nb_E = self.get_neighbor(queries[:, 0])
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