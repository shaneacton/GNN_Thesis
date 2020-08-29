import torch
from torch import nn
import torch.nn.functional as F


class LayerModule(nn.Module):

    """represents either a message, update or aggregate components of a GNN layer"""

    def __init__(self, activation_type, dropout_ratio, activation_kwargs=None):
        """
        creates activation, overriding modules responsibility to use it
        """
        nn.Module.__init__(self)
        self.activation = GenericActivation(activation_type, activation_kwargs)
        self.activate = lambda x: self.activation(x)

        self.dropout_ratio = dropout_ratio

    def dropout(self, vector: torch.Tensor):
        return F.dropout(vector, self.dropout_ratio, self.training)


class GenericActivation(nn.Module):
    """
    a wrapper around activations to allow for usage of any act, with any runtime args provided,
    while only needing to providing the vec in the forward
    """

    def __init__(self, activation_type, activation_kwargs=None):
        super().__init__()
        self.activation_kwargs = activation_kwargs if activation_kwargs else {}
        self.activation_type = activation_type

        self.activation = activation_type()

    def forward(self, vec):
        return self.activation(vec, **self.activation_kwargs)
