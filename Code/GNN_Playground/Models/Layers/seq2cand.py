import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear

from Code.GNN_Playground.Models import embedded_size
from Code.GNN_Playground.Models.Layers.attention_flow import AttentionFlow


class Seq2Cand(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.candidates_reduction = Linear(embedded_size, 2*hidden_size)
        self.att_flow_layer = AttentionFlow(2*hidden_size, 2*hidden_size)

    def forward(self, seq, candidates):
        # (batch, num_candidates, seq_lengths)
        candidates = self.candidates_reduction(candidates)
        distribution: torch.Tensor = self.att_flow_layer(candidates, seq)
        # (batch, num_candidates)
        distribution = torch.sum(distribution, dim=2).squeeze().view(distribution.size(0), -1)
        distribution = F.softmax(distribution, dim=1)
        return distribution
