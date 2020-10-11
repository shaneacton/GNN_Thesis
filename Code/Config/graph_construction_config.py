# word types
from Code.Config.config import Config

ENTITY = "entity"
UNIQUE_ENTITY = "unique_entity"
COREF = "coref"

# structure node types
TOKEN = "token"
WORD = "word"
SENTENCE = "sentence"
PARAGRAPH = "paragraph"
DOCUMENT = "document"

LEVELS = [TOKEN, WORD, SENTENCE, PARAGRAPH, DOCUMENT]

# structure connections
CONNECTION_TYPE = "connection_type"
WINDOW = "window"  # fully connects nodes which are within a max distance of each other
SEQUENTIAL = "sequential"  # connects nodes to the next node at its structure level, within optional max distance
WINDOW_SIZE = "window_size"  # an optional arg for both window and seq
GLOBAL = "global"  # connects to all other nodes

# source types
CONTEXT = "context"
QUERY = "query"
CANDIDATE = "candidate"

# query structure
QUERY_TOKEN = QUERY + "_" + TOKEN  # Longformer style query tokens connected to all context tokens
QUERY_WORD = QUERY + "_" + WORD  # connected to context entity nodes of same string values
QUERY_SENTENCE = QUERY + "_" + SENTENCE  # one node for the whole query


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
        self.context_structure_levels = [TOKEN, SENTENCE]

        # how to connect nodes at the same structure level eg token-token or sentence-sentence
        self.structure_connections = {
            TOKEN: {CONNECTION_TYPE: WINDOW, WINDOW_SIZE: 25},
            WORD: {CONNECTION_TYPE: SEQUENTIAL, WINDOW_SIZE: -1},
            SENTENCE: {CONNECTION_TYPE: SEQUENTIAL, WINDOW_SIZE: 5},
            PARAGRAPH: {CONNECTION_TYPE: SEQUENTIAL, WINDOW_SIZE: 5},
            DOCUMENT: {CONNECTION_TYPE: None, WINDOW_SIZE: -1},
        }

        self.extra_nodes = []
        self.fully_connect_query_nodes = False
        self.query_structure_levels = [QUERY_SENTENCE, QUERY_TOKEN]

        self.query_connections = {  # defines how the query nodes connect to the context. [GLOBAL] option
            QUERY_TOKEN: [TOKEN, SENTENCE],
            QUERY_WORD: [SENTENCE],
            QUERY_SENTENCE: [TOKEN, SENTENCE]
        }

        self.use_candidate_nodes = True
        # which context levels to connect to. automatically connected to all query nodes
        self.candidate_connections = [SENTENCE]

        self.context_max_chars = 50
        self.max_edges = 300000

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

    def get_graph_constructor(self):
        from Code.Data.Graph.Contructors.compound_graph_constructor import CompoundGraphConstructor
        constructors = []
        if self.use_tokens:
            from Code.Data.Graph.Contructors.tokens_constructor import TokensConstructor
            constructors.append(TokensConstructor)
        if self.use_words:
            if ENTITY in self.word_nodes:
                from Code.Data.Graph.Contructors.entities_constructor import EntitiesConstructor
                constructors.append(EntitiesConstructor)
            if COREF in self.word_nodes:
                from Code.Data.Graph.Contructors.coreference_constructor import CoreferenceConstructor
                constructors.append(CoreferenceConstructor)
        if len(self.context_structure_levels) != 1:  # if only one level included - no need for hierarchal structure
            from Code.Data.Graph.Contructors.document_structure_constructor import DocumentStructureConstructor
            constructors.append(DocumentStructureConstructor)

        from Code.Data.Graph.Contructors.window_edge_constructor import WindowEdgeConstructor
        constructors.append(WindowEdgeConstructor)

        if len(self.query_structure_levels):
            from Code.Data.Graph.Contructors.query_constructor import QueryConstructor
            constructors.append(QueryConstructor)

        if self.use_candidate_nodes:
            from Code.Data.Graph.Contructors.candidates_constructor import CandidatesConstructor
            constructors.append(CandidatesConstructor)

        cgc = CompoundGraphConstructor(constructors, self)
        return cgc
