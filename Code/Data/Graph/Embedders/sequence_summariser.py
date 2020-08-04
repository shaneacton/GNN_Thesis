from abc import ABC, abstractmethod
from typing import List

import torch
from torch import nn


class SequenceSummariser(nn.Module, ABC):

    """
    a sequence summariser takes in a (batch * seq_len * feature_size) vec
    uses a basic or learned function to map it to a (batch * 1 * feature_size) vec
    """

    def __init__(self):
        super().__init__()
        self.initialised = False

    def forward(self, embedded_sequence: torch.Tensor):
        return self.summarise(embedded_sequence)

    def summarise(self, embedded_sequence: torch.Tensor):
        if not self.initialised:
            feature_size = embedded_sequence.size(2)
            self.init_layers(feature_size)
        return self._summarise(embedded_sequence)

    @abstractmethod
    def _summarise(self, embedded_sequence: torch.Tensor):
        raise NotImplementedError()

    def init_layers(self, feature_size):
        self._init_layers(feature_size)
        self.initialised = True

    @abstractmethod
    def _init_layers(self, feature_size):
        pass