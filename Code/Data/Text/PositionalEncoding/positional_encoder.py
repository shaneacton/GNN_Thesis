from abc import ABC

from torch import nn


class PositionalEncoder(nn.Module, ABC):

    def __init__(self, num_positions):
        super().__init__()
        self.num_positions = num_positions

