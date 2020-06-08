from typing import Union

from Code.Data.Graph.Contructors.coreference_constructor import CoreferenceConstructor
from Code.Data.Graph.Contructors.document_structure_constructor import DocumentStructureConstructor
from Code.Data.Graph.Contructors.entities_constructor import EntitiesConstructor
from Code.Data.Graph.Contructors.graph_constructor import IncompatibleGraphContructionOrder
from Code.Data.Graph.Contructors.passage_constructor import PassageConstructor
from Code.Data.Graph.Contructors.sentence_contructor import SentenceConstructor
from Code.Data.Graph.Edges.document_edge import DocumentEdge
from Code.Data.Graph.Nodes.document_structure_node import DocumentStructureNode
from Code.Data.Graph.Nodes.entity_node import EntityNode
from Code.Data.Graph.context_graph import ContextGraph
from Code.Data.Text.Tokenisation.document_extract import DocumentExtract
from Code.Data.Text.data_sample import DataSample


class DocumentNodeConstructor(DocumentStructureConstructor):
    def append(self, existing_graph: Union[None, ContextGraph], data_sample: DataSample) -> ContextGraph:
        if not existing_graph or EntitiesConstructor not in existing_graph.constructs \
                or CoreferenceConstructor not in existing_graph.constructs:
            raise IncompatibleGraphContructionOrder(existing_graph, self,
                                                    "Entities and corefs must be graphed before doc nodes")

        tok_seq = data_sample.context.token_sequence

        key_type = DocumentExtract.DOC
        value_node_type = DocumentStructureNode
        if PassageConstructor in existing_graph.constructs:
            value_type = DocumentExtract.PASSAGE
            document_extract = tok_seq.passages
        elif SentenceConstructor in existing_graph.constructs:
            value_type = DocumentExtract.SENTENCE
            document_extract=tok_seq.sentences
        else:
            value_type = DocumentExtract.WORD
            document_extract=tok_seq.entities_and_corefs
            value_node_type=EntityNode

        edge_type = DocumentEdge.get_x2y_edge_type(key_type, value_type)

        self.graph_heirarchical_span_seqs(existing_graph, tok_seq, [tok_seq.full_document], document_extract,
                                          edge_type, value_node_type=value_node_type, key_node_subtype=value_type,
                                          value_node_subtype=value_type)

        existing_graph.constructs.append(type(self))
        return existing_graph

