import torch
from torch import nn, Tensor

from Config import config
from Code.Training import device


class PositionalEmbedder(nn.Module):

    def __init__(self, dims=None, max_positions=4050):
        super().__init__()
        if dims is None:
            dims = config.embedded_dims
        self.dims = dims
        self.max_positions = max_positions
        self.positional_embs = nn.Embedding(max_positions, self.dims)

    def get_pos_embs(self, length):
        ids = self.get_safe_pos_ids(length)
        pos_embs = self.positional_embs(ids).view(1, length, -1)
        return pos_embs

    def get_safe_pos_ids(self, length, ) -> Tensor:
        return torch.tensor([i % self.max_positions for i in range(length)]).to(device)