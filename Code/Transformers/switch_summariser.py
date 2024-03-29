from typing import List, Union

import torch
from torch import Tensor, nn
from torch.nn import LayerNorm
from transformers import TokenSpan
import numpy as np

from Code.Training import dev
from Code.Transformers.summariser import Summariser
from Code.Transformers.switch_transformer import SwitchTransformer
from Code.constants import CANDIDATE, ENTITY, DOCUMENT, GLOBAL
from Config.config import conf

NODE_TYPE_MAP = {ENTITY: 0, DOCUMENT: 1, CANDIDATE: 2}


class SwitchSummariser(SwitchTransformer):
    """
        a summarising longformer which is used to map variable length token embeddings for a node,
        into fixed size node embedding
    """

    def __init__(self, intermediate_fac=2, use_summariser_pos_embs=None):
        self.include_global = conf.use_global_summariser
        if use_summariser_pos_embs is None:
            use_summariser_pos_embs = conf.use_summariser_pos_embs
        super().__init__(conf.embedded_dims, conf.num_summariser_layers, types=[ENTITY, DOCUMENT, CANDIDATE], intermediate_fac=intermediate_fac,
                         include_global=self.include_global,
                         use_pos_embeddings=use_summariser_pos_embs)

        if conf.use_layer_norms_b:
            self.norm = LayerNorm(conf.embedded_dims)

        self.add_cls_token = conf.use_summariser_cls_token
        if self.add_cls_token:
            self.cls_embedding = nn.Embedding(1, conf.hidden_size)

    def get_type_tensor(self, type, length):
        return super().get_type_tensor(type, length, NODE_TYPE_MAP)

    def forward(self, vec_or_vecs: Union[List[Tensor], Tensor], _type, spans: List[TokenSpan]=None, return_list=True):
        """
            either one vec shaped (b, seq, f)
            or a vecs list containing (1, seq, f)
            summaries are returned as a (1, f) list or (b, f)

            if spans is not None, it is a list of token index tuples (s,e), one for each vec
            only these subsequences will be summarised

            if spans is none, the full sequences are summarised
        """
        vecs = vec_or_vecs
        if isinstance(vec_or_vecs, Tensor):
            """break it up into a list of vecs"""
            vecs = vec_or_vecs.split(1, dim=0)

        if spans is None:
            spans = [None] * len(vecs)
        extracts = [Summariser.get_vec_extract(v, spans[i]).view(-1, self.hidden_size) for i, v in enumerate(vecs)]
        if self.use_pos_embeddings:
            extracts = [ex + self.pos_embedder.get_pos_embs(ex.size(0), no_batch=True) for ex in extracts]

        batch, masks = Summariser.pad(extracts)
        if self.add_cls_token:
            ids = torch.tensor([0 for _ in range(batch.size(1))]).to(dev()).long()
            falses = torch.tensor([False for _ in range(batch.size(1))]).to(dev()).bool().view(-1, 1)  # not padding
            cls_emb = self.cls_embedding(ids).view(1, batch.size(1), -1)
            batch = torch.cat([cls_emb, batch], dim=0)
            masks = torch.cat([falses, masks], dim=1)

        enc = self.switch_encoder(batch, src_key_padding_mask=masks, type=_type).transpose(0, 1)
        if self.include_global:
            """we combine the type specific summary with the type agnostic summary"""
            ids = np.ones((batch.size(0), batch.size(1)))
            ids = torch.tensor(ids).to(dev()).long()
            type_embs = self.type_embedder(ids)
            assert type_embs.size() == batch.size()
            glob_batch = self.type_emb_norm(batch + type_embs)

            glob_enc = self.switch_encoder(glob_batch, src_key_padding_mask=masks, type=GLOBAL).transpose(0, 1)
            enc += glob_enc
            if conf.use_layer_norms_b:
                enc = self.norm(enc)

        if conf.use_average_summariser:
            num_tokens = enc.size(1)
            summaries = torch.sum(enc, dim=1) / num_tokens  # (ents, hidd)
        else:
            """simply takes the first element"""
            summaries = enc[:, 0, :]  # (b, hidd)
        if return_list:
            return list(summaries.split(dim=0, split_size=1))

        return summaries