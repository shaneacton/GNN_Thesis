import torch
from torch import nn
from torch.nn import TransformerEncoderLayer, LayerNorm, TransformerEncoder


class Coattention(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(hidden_size, 5, hidden_size * 4, 0.1, 'relu')
        encoder_norm = LayerNorm(hidden_size)
        self.encoder = TransformerEncoder(encoder_layer, 2, encoder_norm)
        self.hidden_size = hidden_size

    def forward(self, suport_embedding, query_embedding):
        # print("supp:", suport_embedding.size(), "query:", query_embedding.size())
        full = torch.cat([suport_embedding, query_embedding], dim=1)
        # print("full:", full.size())
        full = self.encoder(full)

        query_aware_context = full[:, :suport_embedding.size(1), :]
        # print("query aware:", query_aware_context.size())

        return query_aware_context