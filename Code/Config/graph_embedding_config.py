from torch import nn

import Code.constants

from Code.Config.config import Config
from Code.constants import SUMMARISER_NAME, HEAD_AND_TAIL_CAT, SELF_ATTENTIVE_POOLING, NUM_LAYERS


class GraphEmbeddingConfig(Config):

    def __init__(self):
        super().__init__()
        self.use_query_aware_context_vectors = False

        self.span_summarisation_methods = {
            Code.constants.WORD: HEAD_AND_TAIL_CAT,
            Code.constants.SENTENCE: {SUMMARISER_NAME: SELF_ATTENTIVE_POOLING, NUM_LAYERS: 2},
            Code.constants.PARAGRAPH: HEAD_AND_TAIL_CAT,
            Code.constants.DOCUMENT: HEAD_AND_TAIL_CAT,

            Code.constants.QUERY_WORD: HEAD_AND_TAIL_CAT,
            Code.constants.QUERY_SENTENCE: {SUMMARISER_NAME: SELF_ATTENTIVE_POOLING, NUM_LAYERS: 2}
        }

        self.relative_embeddings_window_per_level = {
            Code.constants.TOKEN: 20,
            Code.constants.WORD: 10,
            Code.constants.SENTENCE: 5,
            Code.constants.PARAGRAPH: 3,

            Code.constants.QUERY_TOKEN: 20,
            Code.constants.QUERY_WORD: 10,
            Code.constants.QUERY_SENTENCE: 5,
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
        graph_embedder = GraphEmbedder(self)

        from Code.Data.Text.token_sequence_embedder import TokenSequenceEmbedder
        from Code.Data.Text.pretrained_token_sequence_embedder import tokseq_embedder

        token_embedder: TokenSequenceEmbedder = TokenSequenceEmbedder(self, token_embedder=tokseq_embedder())

        graph_embedder.token_embedder = token_embedder

        from Code.Config import GraphConstructionConfig
        gcc: GraphConstructionConfig = gcc
        for structure_level in gcc.all_structure_levels:
            if structure_level == Code.constants.TOKEN or structure_level == Code.constants.QUERY_TOKEN:
                continue
            graph_embedder.sequence_summarisers[structure_level] = self.get_sequence_embedder(structure_level)

        graph_embedder.on_create_finished()
        return graph_embedder

    def get_sequence_embedder(self, structure_level):
        method_conf = self.span_summarisation_methods[structure_level]
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