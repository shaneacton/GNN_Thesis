from typing import Union

from Code.Data.Graph.Contructors.coreference_constructor import CoreferenceConstructor
from Code.Data.Graph.Contructors.document_structure_constructor import DocumentStructureConstructor
from Code.Data.Graph.Contructors.entities_constructor import EntitiesConstructor
from Code.Data.Graph.Contructors.graph_constructor import IncompatibleGraphContructionOrder
from Code.Data.Graph.Contructors.sentence_contructor import SentenceConstructor
from Code.Data.Graph.Edges.document_edge import DocumentEdge
from Code.Data.Graph.Nodes.entity_node import EntityNode
from Code.Data.Graph.context_graph import ContextGraph
from Code.Data.Text.Tokenisation.document_extract import DocumentExtract
from Code.Data.Text.data_sample import DataSample


class PassageConstructor(DocumentStructureConstructor):

    """
    creates nodes which represent entire passage summaries and links them to the entities contained.
    if the given graph contains sentence nodes, these are linked to instead of the entities
    """

    def append(self, existing_graph: Union[None, ContextGraph], data_sample: DataSample) -> ContextGraph:
        if not existing_graph or EntitiesConstructor not in existing_graph.constructs \
                or CoreferenceConstructor not in existing_graph.constructs:
            raise IncompatibleGraphContructionOrder(existing_graph, self,
                                                    "Entities and corefs must be graphed before sentences")

        tok_seq = data_sample.context.token_sequence

        if SentenceConstructor in existing_graph.constructs:
            edge_type = DocumentEdge.get_x2y_edge_type(DocumentExtract.PASSAGE, DocumentExtract.SENTENCE)
            try:
                self.graph_heirarchical_span_seqs(existing_graph, tok_seq, tok_seq.passages, tok_seq.sentences, edge_type)
            except:
                print("\npassages:","\n".join([repr(span) for span in tok_seq.passages]))
                print("\nsentences:","\n".join([repr(span) for span in tok_seq.sentences]))
                raise Exception()

        else:
            edge_type = DocumentEdge.get_x2y_edge_type(DocumentExtract.PASSAGE, DocumentExtract.WORD)
            self.graph_heirarchical_span_seqs(existing_graph, tok_seq, tok_seq.passages,
                                              tok_seq.entities_and_corefs, edge_type, value_node_type=EntityNode)

        existing_graph.constructs.append(type(self))
        return existing_graph
