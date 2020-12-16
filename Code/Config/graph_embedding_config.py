from torch import nn

from Code.Config.config import Config
from Code.constants import SUMMARISER_NAME, HEAD_AND_TAIL_CAT, SELF_ATTENTIVE_POOLING, NUM_LAYERS, CONTEXT, QUERY, \
    TOKEN, WORD, SENTENCE, PARAGRAPH, DOCUMENT, CANDIDATE, NOUN, ENTITY, COREF


class GraphEmbeddingConfig(Config):

    def __init__(self):
        super().__init__()
        self.use_query_aware_context_vectors = False

        # how to reduce the n feature vectors to 1 for each type of node
        self.span_summarisation_methods = {
            CONTEXT: {
                WORD: HEAD_AND_TAIL_CAT,
                SENTENCE: {SUMMARISER_NAME: SELF_ATTENTIVE_POOLING, NUM_LAYERS: 2},
                PARAGRAPH: HEAD_AND_TAIL_CAT,
                DOCUMENT: HEAD_AND_TAIL_CAT
            },
            QUERY: {
                WORD: HEAD_AND_TAIL_CAT,
                SENTENCE: {SUMMARISER_NAME: SELF_ATTENTIVE_POOLING, NUM_LAYERS: 2}
            },
            CANDIDATE: {WORD: HEAD_AND_TAIL_CAT}
        }

        # used for relative positional embeddings
        # self.relative_embeddings_window_per_level = {
        #     CONTEXT: {
        #         TOKEN: 20,
        #         WORD: 10,
        #         SENTENCE: 5,
        #         PARAGRAPH: 3,
        #     },
        #     QUERY: {
        #         TOKEN: 20,
        #         WORD: 10
        #     }
        # }

    def get_graph_embedder(self, gcc):
        from Code.Data.Graph.Embedders.graph_embedder import GraphEmbedder
        graph_embedder = GraphEmbedder(self, gcc=gcc)

        from Code.Config import GraphConstructionConfig
        gcc: GraphConstructionConfig = gcc
        ss = graph_embedder.sequence_summarisers

        for source in [CONTEXT, QUERY]:
            for structure_level in gcc.structure_levels[source]:
                if structure_level == TOKEN:  # no summary needed for tokens
                    continue
                if source not in ss:
                    ss[source] = {}
                if structure_level in [NOUN, ENTITY, COREF]:
                    """All word levels get same summariser"""
                    structure_level = WORD
                ss[source][structure_level] = self.get_new_sequence_embedder(structure_level, source)

        ss[CANDIDATE] = {}
        ss[CANDIDATE][WORD] = self.get_new_sequence_embedder(WORD, CANDIDATE)

        # print("creating graph embedder with:", graph_embedder.sequence_summarisers)

        graph_embedder.on_create_finished()  # registers summariser params
        return graph_embedder

    def get_new_sequence_embedder(self, structure_level, source):
        method_conf = self.span_summarisation_methods[source][structure_level]
        if isinstance(method_conf, str):
            method = method_conf
        else:
            method = method_conf[SUMMARISER_NAME]

        if method == HEAD_AND_TAIL_CAT:
            from Code.Data.Graph.Embedders.Summarisers.head_and_tail_cat import HeadAndTailCat
            return HeadAndTailCat([], nn.ReLU, 0)
        if method == SELF_ATTENTIVE_POOLING:
            from Code.Data.Graph.Embedders.Summarisers.self_attentive_pool import SelfAttentivePool
            return SelfAttentivePool(method_conf[NUM_LAYERS], [], nn.ReLU, 0)