from abc import ABC
from typing import List

from Code.Data.Graph.Contructors.graph_constructor import GraphConstructor
from Code.Data.Graph.Edges.document_edge import DocumentEdge
from Code.Data.Graph.Nodes.document_structure_node import DocumentStructureNode
from Code.Data.Text.Tokenisation.token_span import TokenSpan


class DocumentStructureConstructor(GraphConstructor, ABC):

    """links up nodes"""

    def graph_heirarchical_span_seqs(self, existing_graph, tok_seq, key_spans: List[TokenSpan],
                                     value_spans: List[TokenSpan], edge_type,
                                     key_node_type=DocumentStructureNode,
                                     value_node_type: type = DocumentStructureNode):
        mapping = tok_seq.match_heirarchical_span_seqs(key_spans, value_spans)

        for key in key_spans:
            node = key_node_type(key)

            contained_value_nodes = [value_node_type(val) for val in mapping[key]]
            contained_value_node_ids = existing_graph.add_nodes(contained_value_nodes)

            key_node_id = existing_graph.add_node(node)
            edges = [DocumentEdge(key_node_id, cont_node_id, subtype=edge_type)
                     for cont_node_id in contained_value_node_ids]

            existing_graph.add_edges(edges)