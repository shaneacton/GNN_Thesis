from typing import List

from torch import Tensor
from transformers import TokenSpan

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

    def __init__(self, intermediate_fac=2):
        self.include_global = conf.use_global_summariser
        super().__init__(conf.embedded_dims, types=[ENTITY, DOCUMENT, CANDIDATE], intermediate_fac=intermediate_fac,
                         include_global=self.include_global)

    def get_type_tensor(self, type, length):
        return super().get_type_tensor(type, length, NODE_TYPE_MAP)

    def forward(self, vecs: List[Tensor], _type, spans: List[TokenSpan]=None):
        if spans is None:
            spans = [None] * len(vecs)
        extracts = [Summariser.get_vec_extract(v, spans[i]).view(-1, self.hidden_size) for i, v in enumerate(vecs)]
        # print("switch extracts:", [e.size() for e in extracts])
        batch, masks = Summariser.pad(extracts)
        enc = self.switch_encoder(batch, src_key_padding_mask=masks, type=_type).transpose(0, 1)
        if self.include_global:
            glob_enc = self.switch_encoder(batch, src_key_padding_mask=masks, type=GLOBAL).transpose(0, 1)
            enc += glob_enc
        # print("summ batch:", batch.size(), "num vecs:", len(vecs))
        summaries = enc[:, 0, :]  # (ents, hidd)
        summaries = summaries.split(dim=0, split_size=1)
        return list(summaries)