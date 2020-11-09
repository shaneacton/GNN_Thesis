from torch import nn

from Code.Config.config import Config
from Code.constants import SUMMARISER_NAME, HEAD_AND_TAIL_CAT, SELF_ATTENTIVE_POOLING, NUM_LAYERS, CONTEXT, QUERY, \
    TOKEN, WORD, SENTENCE, PARAGRAPH, DOCUMENT


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
            }
        }

        # used for relative positional embeddings
        self.relative_embeddings_window_per_level = {
            CONTEXT: {
                TOKEN: 20,
                WORD: 10,
                SENTENCE: 5,
                PARAGRAPH: 3,
            },
            QUERY: {
                TOKEN: 20,
                WORD: 10
            }
        }

        self.token_embedder_type = "bert"
        self.use_contextual_embeddings = True
        self.fine_tune_token_embedder = False

        self.use_absolute_positional_embeddings = True
        self.num_positional_embeddings = 5000

        self.max_bert_token_sequence = 500
        self.bert_window_overlap_tokens = 20
        self.max_token_embedding_threads = 4

    def get_graph_embedder(self, gcc):
        from Code.Data.Graph.Embedders.graph_embedder import GraphEmbedder
        graph_embedder = GraphEmbedder(self, gcc=gcc)

        from Code.Config import GraphConstructionConfig
        gcc: GraphConstructionConfig = gcc
        ss = graph_embedder.sequence_summarisers
        for structure_level in gcc.context_structure_levels:
            if structure_level == TOKEN:  # no summary needed for tokens
                continue
            if CONTEXT not in ss:
                ss[CONTEXT] = {}
            ss[CONTEXT][structure_level] = self.get_sequence_embedder(structure_level, CONTEXT)

        for structure_level in gcc.query_structure_levels:
            if structure_level == TOKEN:  # no summary needed for tokens
                continue
            if QUERY not in ss:
                ss[QUERY] = {}
            ss[QUERY][structure_level] = self.get_sequence_embedder(structure_level, QUERY)

        graph_embedder.on_create_finished()  # registers summariser params
        return graph_embedder

    def get_sequence_embedder(self, structure_level, source):
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