import torch
from torch import nn
from torch.nn import LayerNorm

from Config.config import conf


class HDEScorer(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()

        self.linear1 = nn.Linear(hidden_size, hidden_size//2)
        self.linear2 = nn.Linear(hidden_size//2, 1)
        if conf.use_layer_norms_b:
            self.norm = LayerNorm(hidden_size//2)

    def forward(self, vec):
        vec = torch.tanh(self.linear1(vec))
        if conf.use_layer_norms_b:
            vec = self.norm(vec)
        return torch.tanh(self.linear2(vec))