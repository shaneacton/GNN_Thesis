from abc import ABC, abstractmethod
from typing import List

import torch
from torch import nn


class SequenceSummariser(nn.Module, ABC):

    def forward(self, embedded_sequence: List[torch.Tensor]):
        return self.summarise(embedded_sequence)

    @abstractmethod
    def summarise(self, embedded_sequence: List[torch.Tensor]):
        raise NotImplementedError()