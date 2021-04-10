from typing import Tuple, List

from Code.Embedding.Glove.glove_utils import get_glove_entity_token_spans
from Code.HDE.Graph.graph_utils import get_transformer_entity_token_spans
from Config.config import conf, get_config


class Wikipoint:

    def __init__(self, example, glove_embedder=None, tokeniser=None):
        supports = example["supports"]
        supports = [s[:conf.max_context_chars] if conf.max_context_chars != -1 else s for s in supports]

        self.supports = supports
        self.answer = example["answer"]
        self.candidates = example["candidates"]
        self.query = example["query"]

        if glove_embedder is not None:
            self.ent_token_spans: List[List[Tuple[int]]] = get_glove_entity_token_spans(self, glove_embedder)
        else:
            supp_encs = [tokeniser(supp) for supp in supports]
            self.ent_token_spans: List[List[Tuple[int]]] = get_transformer_entity_token_spans(supp_encs, supports)

    @property
    def relation(self):
        return self.query.split(" ")[0]

    @property
    def query_subject(self):
        return " ".join(self.query.split(" ")[1:])  # the rest

    def __repr__(self):
        ex = "Wiki example:\n"
        ex += "Query:" + self.query + "\n"
        ex += "Candidates:" + ", ".join(self.candidates) + "\n\n"
        ex += "Answer:" + self.answer + "\n"
        ex += "Supports:\n" + "\n\t".join(self.supports)
        return ex

