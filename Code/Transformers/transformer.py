import torch
from torch import nn
from torch.nn import TransformerEncoderLayer, LayerNorm, TransformerEncoder
from torch.nn.utils.rnn import pad_sequence

from Code.Training import device
from Config.config import conf


class Transformer(nn.Module):

    def __init__(self, hidden_size, num_types, num_layers, intermediate_fac=2, use_type_embeddings=True):
        super().__init__()
        self.num_heads = conf.heads
        self.use_type_embeddings = use_type_embeddings
        self.num_types = num_types
        self.hidden_size = hidden_size
        self.type_embedder = nn.Embedding(num_types, self.hidden_size)

        encoder_layer = TransformerEncoderLayer(self.hidden_size, conf.transformer_heads,
                                                self.hidden_size * intermediate_fac, conf.dropout, 'relu')
        encoder_norm = LayerNorm(self.hidden_size)
        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)

    @staticmethod
    def get_type_ids(type, length, type_map):
        type_id = type_map[type]
        type_ids = torch.tensor([type_id for _ in range(length)]).long().to(device())
        return type_ids

    def get_type_tensor(self, type, length, type_map):
        ids = Transformer.get_type_ids(type, length, type_map)
        return self.type_embedder(ids).view(1, -1, self.hidden_size)

    @staticmethod
    def pad(extracts):
        """
            pytorches transformer layer wants 1=pad, 0=seq
            it also wants (seq, batch, emb)
        """
        lengths = [ex.size(-2) for ex in extracts]
        max_len = max(lengths)
        masks = [[False] * size + [True] * (max_len - size) for size in lengths]
        masks = torch.tensor(masks).to(device())
        batch = pad_sequence(extracts, batch_first=False)

        # print("mask:", masks.size(), masks)
        # print("batch:", batch.size())
        return batch, masks