import torch
from torch import nn
from torch.nn import TransformerEncoderLayer, LayerNorm, TransformerEncoder

from Code.Training import device


class Coattention(nn.Module):

    def __init__(self, hidden_size, num_transformer_layers=1, use_type_embeddings=True, num_heads=6, **kwargs):
        super().__init__()

        self.use_type_embeddings = use_type_embeddings
        encoder_layer = TransformerEncoderLayer(hidden_size, num_heads, hidden_size * 2, 0.1, 'relu')
        encoder_norm = LayerNorm(hidden_size)
        self.encoder = TransformerEncoder(encoder_layer, num_transformer_layers, encoder_norm)
        self.hidden_size = hidden_size
        self.type_embedder = nn.Embedding(2, hidden_size)

    def forward(self, suport_embedding, query_embedding):
        """
            (batch, seq, size)
            adds a type embedding to supp and query embeddings
            passes through transformer, returns a transformed sequence shaped as the sup emb
        """
        if self.use_type_embeddings:
            supp_idxs = torch.zeros(suport_embedding.size(1)).long().to(device)
            query_idxs = torch.ones(query_embedding.size(1)).long().to(device)
            suport_embedding += self.type_embedder(supp_idxs).view(1, -1, self.hidden_size)
            query_embedding += self.type_embedder(query_idxs).view(1, -1, self.hidden_size)
        full = torch.cat([suport_embedding, query_embedding], dim=1)
        # print("full:", full.size())
        full = self.encoder(full)

        query_aware_context = full[:, :suport_embedding.size(1), :]
        # print("query aware:", query_aware_context.size())

        return query_aware_context