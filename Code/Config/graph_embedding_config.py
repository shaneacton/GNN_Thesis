from Code.Config import graph_construction_config as construction

# sequence summary functions
HEAD_AND_TAIL_CAT = "head_and_tail_cat"
SELF_ATTENTIVE_POOLING = "self_attentive_pooling"


class GraphEmbeddingConfig:

    def __init__(self):
        self.use_query_aware_context_vectors = False

        self.span_summarisation_methods = {
            construction.WORD: HEAD_AND_TAIL_CAT,
            construction.SENTENCE: HEAD_AND_TAIL_CAT,
            construction.PARAGRAPH: HEAD_AND_TAIL_CAT,
            construction.DOCUMENT: HEAD_AND_TAIL_CAT,

            construction.QUERY_ENTITY: HEAD_AND_TAIL_CAT,
            construction.QUERY_SENTENCE: HEAD_AND_TAIL_CAT
        }

    def get_graph_embedder(self, gcc):
        from Code.Data.Graph.Embedders.graph_embedder import GraphEmbedder
        graph_embedder = GraphEmbedder(self)

        from Code.Data import token_embedder
        from Code.Data.Graph.Embedders.token_embedder import TokenSequenceEmbedder
        token_embedder: TokenSequenceEmbedder = TokenSequenceEmbedder(token_embedder=token_embedder)

        graph_embedder.token_embedder = token_embedder

        from Code.Config import GraphConstructionConfig
        gcc: GraphConstructionConfig = gcc
        for structure_level in gcc.all_structure_levels:
            if structure_level == construction.TOKEN or structure_level == construction.QUERY_TOKEN:
                continue
            graph_embedder.sequence_summarisers[structure_level] = self.get_sequence_embedder(structure_level)

        graph_embedder.on_create_finished()
        return graph_embedder

    def get_sequence_embedder(self, structure_level):
        method = self.span_summarisation_methods[structure_level]
        if method == HEAD_AND_TAIL_CAT:
            from Code.Data.Graph.Embedders.head_and_tail_cat import HeadAndTailCat
            return HeadAndTailCat()
        if method == SELF_ATTENTIVE_POOLING:
            from Code.Data.Graph.Embedders.self_attentive_pool import SelfAttentivePool
            return SelfAttentivePool()