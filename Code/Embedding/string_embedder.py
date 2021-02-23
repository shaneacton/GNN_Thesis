from abc import abstractmethod

from torch import nn


class StringEmbedder(nn.Module):

    @abstractmethod
    def embed(self, string):
        pass

    def forward(self, string):
        return self.embed(string)