import time
from typing import List, Union

import torch
from torch import Tensor
from transformers import TokenSpan

from Code.Embedding.gru_contextualiser import GRUContextualiser
from Code.Training.timer import log_time
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

    def __init__(self, intermediate_fac=2):
        num_types = 4
        use_types = hasattr(conf, "use_summariser_types") and conf.use_summariser_types  # todo remove legacy
        super().__init__(conf.hidden_size, num_types, conf.num_summariser_layers,
                         use_type_embeddings=use_types, intermediate_fac=intermediate_fac)

        if conf.use_switch_coattention:
            self.coattention = SwitchCoattention(intermediate_fac)
        else:
            self.coattention = Coattention(intermediate_fac)

        if not conf.use_simple_hde:
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

        extracts = [self.get_vec_extract(v, spans[i]).view(-1, conf.embedded_dims) for i, v in enumerate(vecs)]
        original_extracts = extracts

        extracts = self.coattention.batched_coattention(extracts, _type, query_vec)
        if not conf.use_simple_hde:  # simple hde skips the gru's and doesn't double model dimensions
            gru_t = time.time()
            extracts = [self.coattention_gru(e) for e in extracts]
            # doubles embedded dim to get hidden dim
            extracts = [torch.cat([e, original_extracts[i]], dim=-1) for i, e in enumerate(extracts)]
            log_time("GRUs", time.time()-gru_t, increment_counter=False)  # signals end of example, needed multiple calls

        batch, masks = self.pad(extracts)
        batch = self.encoder(batch, src_key_padding_mask=masks).transpose(0, 1)
        summaries = batch[:, 0, :]  # (ents, hidd)

        if return_list:
            return list(summaries.split(dim=0, split_size=1))

        return summaries