from typing import List, Union

import torch
from torch import Tensor
from torch.nn import LayerNorm
from transformers import TokenSpan

from Code.Embedding.gru_contextualiser import GRUContextualiser
from Code.Transformers.coattention import Coattention
from Code.Transformers.switch_coattention import SwitchCoattention
from Code.Transformers.transformer import Transformer
from Code.constants import CANDIDATE, ENTITY, DOCUMENT
from Config.config import conf

NODE_TYPE_MAP = {ENTITY: 0, DOCUMENT: 1, CANDIDATE: 2}


class Summariser(Transformer):
    """
        a summarising transformer which is used to map variable length token embeddings for a node,
        into fixed size node embedding.

        here the 3 types are the node types {entity, document, candidate}
    """

    def __init__(self, intermediate_fac=2, use_type_embeddings=True, use_summariser_pos_embs=None):
        num_types = 3
        if use_summariser_pos_embs is None:
            use_summariser_pos_embs = conf.use_summariser_pos_embs
        # todo remove
        if hasattr(conf, "use_summary_coattention") and conf.use_summary_coattention:
            use_type_embeddings = False  # no type embs neccesary, we have distinct params per type instead

        super().__init__(conf.embedded_dims, num_types, conf.num_summariser_layers,
                         use_type_embeddings=use_type_embeddings, intermediate_fac=intermediate_fac,
                         use_pos_embeddings=use_summariser_pos_embs)
        if conf.use_layer_norms_b:
            self.norm = LayerNorm(conf.embedded_dims)

        #todo remove
        if hasattr(conf, "use_summary_coattention") and conf.use_summary_coattention:
            self.coattention = SwitchCoattention(intermediate_fac)

        #todo remove
        if hasattr(conf, "use_coattention_gru") and conf.use_coattention_gru:
            self.coattention_gru = GRUContextualiser()

    def get_type_tensor(self, type, length):
        return super().get_type_tensor(type, length, NODE_TYPE_MAP)

    @staticmethod
    def get_vec_extract(full_vec, span):
        if span is None:
            span = (0, full_vec.size(-2))  # full span
        vec = full_vec[:, span[0]: span[1], :].clone()
        return vec

    def forward(self, vec_or_vecs: Union[List[Tensor], Tensor], _type, spans: List[TokenSpan]=None,
                return_list=True, query_vec: Tensor = None):
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
            """break it up into a list of vecs (1, seq, f)"""
            vecs = vec_or_vecs.split(1, dim=0)

        if spans is None:
            spans = [None] * len(vecs)

        h_size = self.hidden_size
        extracts = [self.get_vec_extract(v, spans[i]).view(-1, h_size) for i, v in enumerate(vecs)]

        # todo remove
        if hasattr(conf, "use_summary_coattention") and conf.use_summary_coattention:
            extracts = self.coattention.batched_coattention(extracts, _type, query_vec)
            if hasattr(conf, "use_coattention_gru") and conf.use_coattention_gru:
                extracts = [self.coattention_gru(e) for e in extracts]

        elif self.use_type_embeddings:
            extracts = [ex + self.get_type_tensor(_type, ex.size(-2)).view(-1, h_size) for ex in extracts]

        if self.use_pos_embeddings:
            extracts = [ex + self.pos_embedder.get_pos_embs(ex.size(0), no_batch=True) for ex in extracts]
        if conf.use_layer_norms_b:
            extracts = [self.norm(ex) for ex in extracts]

        original_batch, masks = self.pad(extracts)
        batch = self.encoder(original_batch, src_key_padding_mask=masks).transpose(0, 1)
        if hasattr(conf, "use_concat_summaries") and conf.use_concat_summaries:
            batch = torch.cat([original_batch.transpose(0, 1), batch], dim=-1)

        if conf.use_average_summariser:
            num_tokens = batch.size(1)
            summaries = torch.sum(batch, dim=1) / num_tokens  # (ents, hidd)
        else:
            summaries = batch[:, 0, :]  # (ents, hidd)

        if return_list:
            return list(summaries.split(dim=0, split_size=1))

        return summaries