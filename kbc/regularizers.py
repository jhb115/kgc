
from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn


class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass


class N2(Regularizer):

    def __init__(self, weight: float):
        super(N2, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(
                torch.norm(f, 2, 1) ** 3
            )
        return norm / factors[0].shape[0]


class N3(Regularizer):

    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(
                torch.abs(f) ** 3
            )
        return norm / factors[0].shape[0]


# For Context model
# factors = [s, r, o, c]
class N4(Regularizer):

    def __init__(self, weight: float, g_weight=0.02):
        super(N4, self).__init__()
        self.weight = weight
        self.g_weight = g_weight

    def forward(self, factors, g=torch.Tensor([0.])):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(
                f ** 3
            )
        norm /= factors[0].shape[0]
        norm += self.g_weight * torch.sum(g ** 2)
        return norm
