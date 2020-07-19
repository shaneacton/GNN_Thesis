# word types

ENTITY = "entity"
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
WINDOW_SIZE = "window_size"

class Configuration:



    def __init__(self):

        self.word_nodes = [ENTITY, COREF]
        self.structure_nodes = [WORD, SENTENCE]  # elements from DocumentExtrac.levels

        self.structure_connections = {
            TOKEN: {CONNECTION_TYPE: WINDOW, WINDOW_SIZE: 20},
            WORD: {CONNECTION_TYPE: SEQUENTIAL, WINDOW_SIZE: -1},
            SENTENCE: {CONNECTION_TYPE: SEQUENTIAL, WINDOW_SIZE: -1},
            PARAGRAPH: {CONNECTION_TYPE: SEQUENTIAL, WINDOW_SIZE: -1},
            DOCUMENT: {CONNECTION_TYPE: None, WINDOW_SIZE: -1},
        }

    def getGraphConstructor(self):
        from Code.Data.Graph.Contructors.compound_graph_constructor import CompoundGraphConstructor
        constructors = []
        if "entity" in self.word_nodes:
            from Code.Data.Graph.Contructors.entities_constructor import EntitiesConstructor
            constructors.append(EntitiesConstructor)
        if "coref" in self.word_nodes:
            print("adding coref const")
            from Code.Data.Graph.Contructors.coreference_constructor import CoreferenceConstructor
            constructors.append(CoreferenceConstructor)
        if len(self.structure_nodes) != 1:  # if only one level included - no need for hierarchal structure
            from Code.Data.Graph.Contructors.document_structure_constructor import DocumentStructureConstructor
            constructors.append(DocumentStructureConstructor)

        from Code.Data.Graph.Contructors.window_edge_constructor import WindowEdgeConstructor
        constructors.append(WindowEdgeConstructor)

        cgc = CompoundGraphConstructor(constructors)
        return cgc
