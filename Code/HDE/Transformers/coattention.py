from typing import List

import torch
from torch import Tensor

from Code.HDE.Transformers.transformer import Transformer
from Code.Training import device
from Code.constants import QUERY, DOCUMENT
from Config.config import conf

SOURCE_TYPE_MAP = {DOCUMENT: 0, QUERY: 1}


class Coattention(Transformer):

    def __init__(self, intermediate_fac=2, use_type_embedder=True):
        super().__init__(conf.embedded_dims, 2, use_type_embeddings=use_type_embedder, intermediate_fac=intermediate_fac)

    def get_type_tensor(self, type, length):
        return super().get_type_tensor(type, length, SOURCE_TYPE_MAP)

    def batched_coattention(self, supps: List[Tensor], query: Tensor):
        if self.use_type_embeddings:
            supps = [s + self.get_type_tensor(DOCUMENT, s.size(-2)) for s in supps]
            query = (query + self.get_type_tensor(QUERY, query.size(-2)))

        supps = [s.view(-1, self.hidden_size) for s in supps]
        query = query.view(-1, self.hidden_size)

        cats = [torch.cat([supp, query], dim=0) for supp in supps]
        batch, masks = self.pad(cats)
        # print("coat batch:", batch.size(), "num supps:", len(supps))
        batch = self.encoder(batch, src_key_padding_mask=masks).transpose(0, 1)
        seqs = list(batch.split(dim=0, split_size=1))
        assert len(seqs) == len(supps)
        for s, seq in enumerate(seqs):
            seqs[s] = seq[:, :supps[s].size(0), :]
        # print("seqs:", [s.size() for s in seqs])
        return seqs

    def forward(self, supps: List[Tensor], query: Tensor):
        return self.batched_coattention(supps, query)