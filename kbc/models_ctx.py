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

class Context_ComplEx(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int, sorted_data:np.ndarray,
            slice_dic: np.ndarray, max_NB: int=50, init_size: float=1e-3,
            data_name: str='FB15K'
    ):
        super(Context_ComplEx, self).__init__()
        n_s, n_r, n_o = sizes
        self.sizes = [n_s, n_r, n_o, n_o]  #append another n_o for nb_o
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 3 * rank, sparse=True)
            for s in self.sizes[:3]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size  # For context

        self.W = torch.randn((rank*2, rank)).cuda(), torch.randn((rank*2, rank)).cuda()
        self.b_w = torch.randn((1, rank)).cuda(), torch.randn((1, rank)).cuda()  # bias term
        self.W2 = torch.randn((rank, rank)).cuda(), torch.randn((rank, rank)).cuda()
        self.b_w2 = torch.randn((1, rank)).cuda(), torch.randn((1, rank)).cuda()

        nn.init.xavier_uniform_(self.W[0])
        nn.init.xavier_uniform_(self.W2[0])
        nn.init.xavier_uniform_(self.W[1])
        nn.init.xavier_uniform_(self.W2[1])

        nn.init.xavier_uniform_(self.b_w[0])
        nn.init.xavier_uniform_(self.b_w[1])
        nn.init.xavier_uniform_(self.b_w2[0])
        nn.init.xavier_uniform_(self.b_w2[1])

        self.drop_layer1 = nn.Dropout(p=0.3)
        self.drop_layer2 = nn.Dropout(p=0.3)

        self.Wo = torch.randn((rank, 1)).cuda(), torch.randn((rank, 1)).cuda()
        self.b_wo = torch.randn((1, 1)).cuda(), torch.randn((1, 1)).cuda()
        self.Uo = torch.randn((rank, 1)).cuda(), torch.randn((rank, 1)).cuda()
        self.b_uo = torch.randn((1, 1)).cuda(), torch.randn((1, 1)).cuda()

        nn.init.xavier_uniform_(self.Wo[0])
        nn.init.xavier_uniform_(self.Uo[0])
        nn.init.xavier_uniform_(self.Wo[1])
        nn.init.xavier_uniform_(self.Uo[1])

        nn.init.xavier_uniform_(self.b_wo[0])
        nn.init.xavier_uniform_(self.b_wo[1])
        nn.init.xavier_uniform_(self.b_uo[0])
        nn.init.xavier_uniform_(self.b_uo[1])

        self.sorted_data = sorted_data
        self.slice_dic = slice_dic
        self.max_NB = max_NB

        self.forward_g = []
        self.valid_g = []

        self.i = 0

    def get_neighbor(self, subj: torch.Tensor):
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

        return self.embeddings[2](index_tensor)

    def score(self, x: torch.Tensor):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        # Concatenation of lhs, rel
        trp_E = torch.cat((lhs[0], rel[0]), dim=1), torch.cat((lhs[1], rel[1]), dim=1)

        # Get attention weight vector, linear projection of trp_E
        # w = self.W(trp_E)
        w_R = torch.einsum('', self.W[0], trp_E[0]) - torch.einsum(''. self.W[1], self.trp_E[1]) + self.b_w[0]
        w_I = torch.einsum('', self.W[1], trp_E[0]) + torch.einsum('', self.W[0], trp_E[1]) + self.b_w[1]

        w = w_R, w_I

        nb_E = self.get_neighbor(x[:, 0])
        nb_E = nb_E[:, :, :self.rank], nb_E[:, :, self.rank:]  # check on this

        # Take the real part of w @ nb_E
        alpha = torch.softmax(torch.einsum('', w[0], nb_E[0]) - torch.einsum('', w[1], nb_E[1]), dim=1)

        # e_c = self.W2(torch.einsum('bm,bmk->bk', alpha, nb_E))
        e_c = torch.einsum('', alpha, nb_E[0]), torch.einsum('', alpha, nb_E[1])

        # Linear matrix multiplication
        e_c_R = torch.einsum('', e_c[0], self.W2[0]) - torch.einsum('', e_c[1], self.W2[1]) + self.b_w2[0]
        e_c_I = torch.einsum('', e_c[0], self.W2[1]) + torch.einsum('', e_c[1], self.W2[0]) + self.b_w2[1]

        e_c = e_c_R, e_c_I

        # calculation of g
        g = Sigmoid(torch.einsum('', self.Uo[0], lhs[0]*rel[0]-lhs[1]*rel[1])
                    - torch.einsum('', self.Uo[1], lhs[1]*rel[0]+lhs[0]*rel[1]))

        gated_e_c = g * e_c[0] + (torch.ones((self.chunk_size, 1)).cuda() - g)*torch.ones_like(e_c[0]), g * e_c[1]

        rror_rioi = rel[0]*rhs[0]+rel[1]*rhs[1]
        rior = rel[1]*rhs[0]
        rroi = rel[0]*rhs[1]

        return torch.sum((lhs[0]*rror_rioi + lhs[1]*(rior + rroi))*gated_e_c[0]
                          + (lhs[1]*rror_rioi + lhs[0]*(rior - rroi))*gated_e_c[1], 1, keepdim=True)























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






















