from Code.Config.config import Config
from Code.constants import ENTITY, COREF, TOKEN, WORD, SENTENCE, PARAGRAPH, DOCUMENT, CONNECTION_TYPE, WINDOW, \
    SEQUENTIAL, WINDOW_SIZE


class GraphConstructionConfig(Config):

    def __init__(self):

        super().__init__()
        # {ENTITY, COREF, UNIQUE_ENTITY}
        self.word_nodes = [ENTITY]  # types of word nodes to use

        # empty for no filters. filters us OR not AND when combined.
        # This means filters [CANDIDATE, QUERY] allows which are either candidates or queries
        self.word_node_filters = []

        # which structure levels to make nodes for
        # {TOKEN, WORD, SENTENCE, PARAGRAPH, DOCUMENT}
        self.context_structure_levels = [TOKEN, WORD, SENTENCE]

        # how to connect nodes at the same structure level eg token-token or sentence-sentence
        self.structure_connections = {
            TOKEN: {CONNECTION_TYPE: WINDOW, WINDOW_SIZE: 25},
            WORD: {CONNECTION_TYPE: SEQUENTIAL, WINDOW_SIZE: -1},
            SENTENCE: {CONNECTION_TYPE: SEQUENTIAL, WINDOW_SIZE: 5},
            PARAGRAPH: {CONNECTION_TYPE: SEQUENTIAL, WINDOW_SIZE: 5},
            DOCUMENT: {CONNECTION_TYPE: None, WINDOW_SIZE: -1},
        }

        self.extra_nodes = []
        self.fully_connect_query_nodes = True
        self.query_structure_levels = [TOKEN, WORD, SENTENCE]

        self.query_connections = {  # defines how the query nodes connect to the context. [GLOBAL] option
            TOKEN: [TOKEN, SENTENCE],
            WORD: [SENTENCE],
            SENTENCE: [TOKEN, SENTENCE]
        }

        self.use_candidate_nodes = True
        # which context levels to connect to. automatically connected to all query nodes
        self.candidate_connections = [SENTENCE]

        self.max_edges = 400000  # 400000
        # cuts off all chars after the max. tries to proceed with partial context. -1 to turn off
        self.max_context_chars = 1000

    @property
    def all_structure_levels(self):
        levels = self.context_structure_levels + self.query_structure_levels
        if self.use_candidate_nodes:
            levels.append(WORD)
        return set(levels)

    def has_keyword(self, word:str):
        return word in self.word_nodes or word in self.context_structure_levels or word in self.extra_nodes

    @property
    def use_words(self):
        return WORD in self.context_structure_levels

    @property
    def use_tokens(self):
        return TOKEN in self.context_structure_levels