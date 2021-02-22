from typing import Tuple, List

from Code.Config.config import config
from Code.Embedding.Glove.glove_utils import get_glove_entity_token_spans


class Wikipoint:

    def __init__(self, example, glove_embedder=None):
        supports = example["supports"]
        supports = [s[:config.max_context_chars] if config.max_context_chars != -1 else s for s in supports]

        self.ent_token_spans: List[List[Tuple[int]]] = get_glove_entity_token_spans(supports, glove_embedder)

        self.supports = supports
        self.answer = example["answer"]
        self.candidates = example["candidates"]
        self.query = example["query"]

        # graph = HDEGloveStack.create_graph(self.candidates, self.ent_token_spans, supports, glove_embedder)
        # self.edge_list = graph.edge_list
