from typing import List

import torch
from torch import Tensor

from Code.HDE.Transformers.transformer import Transformer
from Code.Training import device
from Code.constants import QUERY, DOCUMENT

SOURCE_TYPE_MAP = {DOCUMENT: 0, QUERY: 1}


class Coattention(Transformer):

    def __init__(self, hidden_size, num_transformer_layers=1, num_heads=6, intermediate_fac=2, dropout=0.1,
                 use_type_embedder=True, **kwargs):
        super().__init__(hidden_size, 2, use_type_embeddings=use_type_embedder, intermediate_fac=intermediate_fac,
                         dropout=dropout, num_transformer_layers=num_transformer_layers, num_heads=num_heads)

    def get_type_tensor(self, type, length):
        return super().get_type_tensor(type, length, SOURCE_TYPE_MAP)

    def batched_coattention(self, supps: List[Tensor], query: Tensor):
        if self.use_type_embeddings:
            supps = [s + self.get_type_tensor(DOCUMENT, s.size(-2)) for s in supps]
            query = (query + self.get_type_tensor(QUERY, query.size(-2)))

        supps = [s.view(-1, self.hidden_size) for s in supps]
        query = query.view(-1, self.hidden_size)

        # print("supps:", [s.size() for s in supps])
        # print("query:", query.size())
        cats = [torch.cat([supp, query], dim=0) for supp in supps]
        # print("cats:", [c.size() for c in cats])
        batch, masks = self.pad(cats)
        batch = self.encoder(batch, src_key_padding_mask=masks).transpose(0, 1)
        # print("batch:", batch.size())
        seqs = list(batch.split(dim=0, split_size=1))
        assert len(seqs) == len(supps)
        for s, seq in enumerate(seqs):
            seqs[s] = seq[:, :supps[s].size(0), :]
        # print("seqs:", [s.size() for s in seqs])
        return seqs

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