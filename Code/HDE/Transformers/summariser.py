from typing import List

from torch import Tensor
from transformers import TokenSpan

from Config import config
from Code.HDE.Transformers.transformer import Transformer
from Code.constants import CANDIDATE, ENTITY, DOCUMENT

NODE_TYPE_MAP = {ENTITY: 0, DOCUMENT: 1, CANDIDATE: 2}


class Summariser(Transformer):
    """
        a summarising longformer which is used to map variable length token embeddings for a node,
        into fixed size node embedding
    """

    def __init__(self, intermediate_fac=2, use_type_embeddings=True):
        super().__init__(config.embedded_dims, num_types=3, use_type_embeddings=use_type_embeddings, intermediate_fac=intermediate_fac)

    def get_type_tensor(self, type, length):
        return super().get_type_tensor(type, length, NODE_TYPE_MAP)

    @staticmethod
    def get_vec_extract(full_vec, span):
        if span is None:
            span = (0, full_vec.size(-2))  # full span
        vec = full_vec[:, span[0]: span[1], :].clone()
        return vec

    def forward(self, vecs: List[Tensor], _type, spans: List[TokenSpan]=None):
        if spans is None:
            spans = [None] * len(vecs)
        extracts = [self.get_vec_extract(v, spans[i]) for i, v in enumerate(vecs)]
        if self.use_type_embeddings:
            extracts = [(ex + self.get_type_tensor(_type, ex.size(1))).view(-1, self.hidden_size) for ex in extracts]
        # print("before:", [ex.size() for ex in extracts])
        batch, masks = self.pad(extracts)
        # print("batch:", batch.size())
        batch = self.encoder(batch, src_key_padding_mask=masks).transpose(0, 1)
        summaries = batch[:, 0, :]  # (ents, hidd)
        # print("summs:", summaries.size())
        summaries = summaries.split(dim=0, split_size=1)
        # summaries = [s.view(1, 1, -1) for s in summaries]
        # print("split sums:", [s.size() for s in summaries], type(summaries))
        return list(summaries)