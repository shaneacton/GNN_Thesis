from abc import ABC, abstractmethod
from typing import List

import torch
from torch import nn


class SequenceSummariser(nn.Module, ABC):

    """
    a sequence summariser takes in a (batch * seq_len * feature_size) vec
    uses a basic or learned function to map it to a (batch * 1 * feature_size) vec
    """

    def forward(self, embedded_sequence: List[torch.Tensor]):
        return self.summarise(embedded_sequence)

    @abstractmethod
    def summarise(self, embedded_sequence: List[torch.Tensor]):
        raise NotImplementedError()