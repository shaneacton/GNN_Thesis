# word types
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

# query stucture
QUERY_TOKEN = "query_token"  # Longformer style query tokens connected to all context tokens
QUERY_ENTITY = "query_entity"  # connected to context entity nodes of same string values
QUERY_SENTENCE = "query_sentence"  # one node for the whole query

# source types
CONTEXT = "context"
QUERY = "query"
CANDIDATE = "candidate"


class GraphConstructionConfig:

    def __init__(self):

        self.word_nodes = [ENTITY, COREF]  # types of word nodes to use
        # empty for no filters. filters us OR not AND when combined.
        # This means filters [CANDIDATE, QUERY] allows which are either candidates or queries
        self.word_node_filters = []
        self.context_structure_nodes = [TOKEN, WORD, SENTENCE, PARAGRAPH, DOCUMENT]  # which structure levels to make nodes for

        # how to connect nodes at the same structure level eg token-token or sentence-sentence
        self.structure_connections = {
            TOKEN: {CONNECTION_TYPE: SEQUENTIAL, WINDOW_SIZE: 6},
            WORD: {CONNECTION_TYPE: SEQUENTIAL, WINDOW_SIZE: -1},
            SENTENCE: {CONNECTION_TYPE: SEQUENTIAL, WINDOW_SIZE: -1},
            PARAGRAPH: {CONNECTION_TYPE: SEQUENTIAL, WINDOW_SIZE: -1},
            DOCUMENT: {CONNECTION_TYPE: None, WINDOW_SIZE: -1},
        }

        self.extra_nodes = []
        self.query_node_types = [QUERY_TOKEN, QUERY_ENTITY, QUERY_SENTENCE]

        self.query_connections = {  # defines how the query nodes connect to the context
            QUERY_TOKEN: [TOKEN],
            QUERY_ENTITY: [WORD],
            QUERY_SENTENCE: [SENTENCE]
        }

        self.context_max_chars = 50

    @property
    def all_structure_levels(self):
        return self.context_structure_nodes + self.query_node_types

    def has_keyword(self, word:str):
        return word in self.word_nodes or word in self.context_structure_nodes or word in self.extra_nodes

    @property
    def use_words(self):
        return WORD in self.context_structure_nodes

    @property
    def use_tokens(self):
        return TOKEN in self.context_structure_nodes

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
        if len(self.context_structure_nodes) != 1:  # if only one level included - no need for hierarchal structure
            from Code.Data.Graph.Contructors.document_structure_constructor import DocumentStructureConstructor
            constructors.append(DocumentStructureConstructor)

        from Code.Data.Graph.Contructors.window_edge_constructor import WindowEdgeConstructor
        constructors.append(WindowEdgeConstructor)

        if len(self.query_node_types):
            from Code.Data.Graph.Contructors.query_constructor import QueryConstructor
            constructors.append(QueryConstructor)

        cgc = CompoundGraphConstructor(constructors)
        return cgc
