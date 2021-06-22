import torch
from torch import nn
from torch.nn import LayerNorm

from Config.config import conf


class HDEScorer(nn.Module):

    def __init__(self, size):
        super().__init__()

        self.linear1 = nn.Linear(size, size//2)
        self.linear2 = nn.Linear(size//2, 1)

    def forward(self, vec):
        vec = torch.tanh(self.linear1(vec))
        return torch.tanh(self.linear2(vec))