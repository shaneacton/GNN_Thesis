import torch
from Code.Data.Text.PositionalEncoding.positional_encoder import PositionalEncoder


class RelativePositionalEncoder(PositionalEncoder):

    def __init__(self, num_positions):
        super().__init__(num_positions)

    def forward(self, edge_index_i: torch.Tensor, edge_index_j: torch.Tensor):
        """
        For GNN emb_seq is (E, features)
        """
        pass
