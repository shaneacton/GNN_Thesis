from typing import Union

from Code.Data.Graph.Contructors.coreference_constructor import CoreferenceConstructor
from Code.Data.Graph.Contructors.document_structure_constructor import DocumentStructureConstructor
from Code.Data.Graph.Contructors.entities_constructor import EntitiesConstructor
from Code.Data.Graph.Contructors.graph_constructor import IncompatibleGraphContructionOrder
from Code.Data.Graph.Edges.document_edge import DocumentEdge
from Code.Data.Graph.Nodes.entity_node import EntityNode
from Code.Data.Graph.context_graph import ContextGraph
from Code.Data.Text.Tokenisation.document_extract import DocumentExtract
from Code.Data.Text.data_sample import DataSample


class SentenceConstructor(DocumentStructureConstructor):

    """
    creates nodes which represent entire sentence summaries and links these nodes to the contained entity mentions and corefs
    """

    def append(self, existing_graph: Union[None, ContextGraph], data_sample: DataSample) -> ContextGraph:
        if not existing_graph or EntitiesConstructor not in existing_graph.constructs \
                or CoreferenceConstructor not in existing_graph.constructs:
            raise IncompatibleGraphContructionOrder(existing_graph, self,
                                                    "Entities and corefs must be graphed before sentences")

        tok_seq = data_sample.context.token_sequence
        edge_type = DocumentEdge.get_x2y_edge_type(DocumentExtract.SENTENCE, DocumentExtract.WORD)
        self.graph_heirarchical_span_seqs(existing_graph, tok_seq, tok_seq.sentences, tok_seq.entities_and_corefs,
                                          edge_type, value_node_type=EntityNode,
                                          key_node_subtype=DocumentExtract.SENTENCE,
                                          value_node_subtype=DocumentExtract.WORD)

        existing_graph.constructs.append(type(self))
        return existing_graph








