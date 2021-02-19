from typing import List

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.nn import TransformerEncoderLayer, LayerNorm, TransformerEncoder
from transformers import LongformerModel, TokenSpan

from Code.Training import device
from Code.constants import CANDIDATE, ENTITY, DOCUMENT

NODE_TYPE_MAP = {ENTITY: 0, DOCUMENT: 1, CANDIDATE: 2}


class Summariser(nn.Module):
    """
        a summarising longformer which is used to map variable length token embeddings for a node,
        into fixed size node embedding
    """

    def __init__(self, hidden_size, num_transformer_layers=1, num_heads=6, intermediate_fac=2, dropout=0.1, use_type_embedder=True, **kwargs):
        super().__init__()
        self.use_type_embedder = use_type_embedder
        encoder_layer = TransformerEncoderLayer(hidden_size, num_heads, hidden_size * intermediate_fac, dropout, 'relu')
        encoder_norm = LayerNorm(hidden_size)
        self.encoder = TransformerEncoder(encoder_layer, num_transformer_layers, encoder_norm)
        self.hidden_size = hidden_size
        self.type_embedder = nn.Embedding(3, hidden_size)  # entity, candidate, document

        nn.Transformer()

    @staticmethod
    def get_type_ids(type, length, type_map=NODE_TYPE_MAP):
        type_id = type_map[type]
        type_ids = torch.tensor([type_id for _ in range(length)]).long().to(device)
        return type_ids

    def get_type_tensor(self, type, length, type_map=NODE_TYPE_MAP):
        ids = Summariser.get_type_ids(type, length, type_map)
        return self.type_embedder(ids).view(1, -1, self.hidden_size)

    @staticmethod
    def get_vec_extract(full_vec, span):
        if span is None:
            span = (0, full_vec.size(-2))  # full span
        vec = full_vec[:, span[0]: span[1], :].clone()
        return vec

    def pad(self, extracts):
        """pytorches transformer layer wants 1=pad, 0=seq"""
        # print("sizes:", [ex.size() for ex in extracts])
        lengths = [ex.size(-2) for ex in extracts]
        # print("lens:", lengths)
        max_len = max(lengths)
        masks = [[0] * size + [1] * (max_len - size) for size in lengths]
        # for size in lengths:
        #     print("size:", size, "comp:", (max_len - size))
            # print([0] * size, [1] * (max_len - size), [0] * size + [1] * (max_len - size))
        masks = torch.tensor(masks).to(device)
        print("mask:", masks.size(), masks)
        batch = pad_sequence(extracts, batch_first=True)
        return batch, masks

    def get_batched_summary_vec(self, vecs: List[Tensor], _type, spans: List[TokenSpan]=None):
        if spans is None:
            spans = [None] * len(vecs)
        extracts = [self.get_vec_extract(v, spans[i]) for i, v in enumerate(vecs)]
        if self.use_type_embedder:
            extracts = [(ex + self.get_type_tensor(_type, ex.size(1))).view(-1, self.hidden_size) for ex in extracts]
        # print("before:", [ex.size() for ex in extracts])
        batch, masks = self.pad(extracts)
        # print("batch:", batch.size())
        batch = self.encoder(batch)
        summaries = batch[:, 0, :]  # (ents, hidd)
        # print("summs:", summaries.size())
        summaries = summaries.split(dim=0, split_size=1)
        # summaries = [s.view(1, 1, -1) for s in summaries]
        # print("split sums:", [s.size() for s in summaries], type(summaries))
        return list(summaries)

    def get_summary_vec(self, full_vec: Tensor, type, span=None):
        """
            if provided span is the [start,end) indices of the full vec which are to be summarised
            :full_vec ~ (batch, seq_len, features)

            returns (batch, features)
        """
        vec = self.get_vec_extract(full_vec, span)
        if self.use_type_embedder:
            vec += self.get_type_tensor(type, vec.size(1))

        embs = self.encoder(vec)
        pooled_emb = embs[:, 0]  # the pooled out is the output of the classification token
        return pooled_emb

    def forward(self, vec_or_vecs: Tensor, type, span_or_spans=None):
        if isinstance(vec_or_vecs, List):
            return self.get_batched_summary_vec(vec_or_vecs, type, span_or_spans)
        return self.get_summary_vec(vec_or_vecs, type, span_or_spans)