from abc import ABC, abstractmethod

from torch import nn, Tensor


class OutputModel(nn.Module, ABC):

    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

    def forward(self, x, **kwargs):
        from Code.Data.Graph.Embedders.graph_encoding import GraphEncoding

        if isinstance(x, GraphEncoding):
            return self.get_output_from_graph_encoding(x, **kwargs)
        if isinstance(x, Tensor):
            return self.get_output_from_tensor(x, **kwargs)

    @abstractmethod
    def get_output_from_graph_encoding(self, data, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_output_from_tensor(self, x: Tensor, **kwargs):
        raise NotImplementedError()