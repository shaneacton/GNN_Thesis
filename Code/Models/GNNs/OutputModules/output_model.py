from abc import ABC

from torch import nn


class OutputModel(nn.Module, ABC):

    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features