from abc import abstractmethod

from torch import nn


class StringEmbedder(nn.Module):

    @abstractmethod
    def embed(self, string, **kwargs):
        pass

    def forward(self, string, **kwargs):
        return self.embed(string, **kwargs)