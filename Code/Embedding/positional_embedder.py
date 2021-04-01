import torch
from torch import nn, Tensor

from Code.Training import dev
from Config.config import conf


class PositionalEmbedder(nn.Module):

    def __init__(self, dims=None, max_positions=4050):
        super().__init__()
        if dims is None:
            dims = conf.embedded_dims
        self.dims = dims
        self.max_positions = max_positions
        self.positional_embs = nn.Embedding(max_positions, self.dims)

    def get_pos_embs(self, length, no_batch=False):
        """returns (1, seq, f) or (seq, f) if no_batch=true"""
        ids = self.get_safe_pos_ids(length)
        if no_batch:
            return self.positional_embs(ids)
        return self.positional_embs(ids).view(1, length, -1)

    def get_safe_pos_ids(self, length, ) -> Tensor:
        return torch.tensor([i % self.max_positions for i in range(length)]).to(dev())