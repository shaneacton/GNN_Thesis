import inspect
from typing import List

import torch
import torch.nn.functional as F
from torch import nn


class GNNComponent:

    def __init__(self, sizes: List[int], activation_type, dropout_ratio, activation_kwargs=None):
        self.sizes = sizes
        self.dropout_ratio = dropout_ratio

        if not activation_type:
            # print("no activation type for", self)
            return

        self.activation = GenericActivation(activation_type, activation_kwargs)
        # print("comp:", self, "act:", self.activation)
        self.activate = lambda x: self.activation(x)

    def dropout(self, vector: torch.Tensor, training):
        return F.dropout(vector, self.dropout_ratio, training)

    @staticmethod
    def get_method_arg_names(method):
        return inspect.getfullargspec(method)[0]

    @staticmethod
    def get_needed_args(accepted_args, available_args):
        """returns all of the available args which are accepted"""
        # print("getting needed args:",accepted_args, "from:",available_args)
        return {arg: available_args[arg] for arg in available_args.keys() if arg in accepted_args}

    # @property
    # def num_params(self):
    #     return sum(p.numel() for p in self.parameters())

    @property
    def input_size(self):
        return self.sizes[0]

    @property
    def channels(self):
        if len(self.sizes) != 1:
            raise Exception("only single sized components have channel count" + repr(self))
        return self.sizes[0]

    @property
    def output_size(self):
        if len(self.sizes) < 2:
            raise Exception("no output size on " + repr(self))
        return self.sizes[-1]

    @property
    def hidden_size(self):
        if len(self.sizes) < 3:
            raise Exception("no hidden size on " + repr(self))
        return self.sizes[1]


class GenericActivation(nn.Module):
    """
    a wrapper around activations to allow for usage of any act, with any runtime args provided,
    while only needing to providing the vec in the forward
    """

    def __init__(self, activation_type, activation_kwargs=None):
        super().__init__()
        self.activation_kwargs = activation_kwargs if activation_kwargs else {}
        self.activation_type = activation_type

        self.activation_function = activation_type()

    def forward(self, vec):
        return self.activation_function(vec, **self.activation_kwargs)