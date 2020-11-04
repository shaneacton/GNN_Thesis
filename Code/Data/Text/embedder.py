from abc import ABC, abstractmethod

from torch import nn
from transformers import BatchEncoding


class Embedder(nn.Module, ABC):

    @abstractmethod
    def embed(self, encoding: BatchEncoding):
        raise NotImplementedError()

    def forward(self, encoding: BatchEncoding):
        return self.embed(encoding)
