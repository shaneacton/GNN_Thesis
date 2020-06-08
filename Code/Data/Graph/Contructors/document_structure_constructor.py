from abc import ABC
from typing import List, Type

from Code.Data.Graph.Contructors.graph_constructor import GraphConstructor
from Code.Data.Graph.Edges.document_edge import DocumentEdge
from Code.Data.Graph.Nodes.document_structure_node import DocumentStructureNode
from Code.Data.Graph.Nodes.node import Node
from Code.Data.Text.Tokenisation.token_span import TokenSpan


class DocumentStructureConstructor(GraphConstructor, ABC):

    """links up nodes"""

    def graph_heirarchical_span_seqs(self, existing_graph, tok_seq, key_spans: List[TokenSpan],
                                     value_spans: List[TokenSpan], edge_type: str,
                                     key_node_type:Type[Node]=DocumentStructureNode, key_node_subtype:str=None,
                                     value_node_type: Type[Node] = DocumentStructureNode, value_node_subtype:str=None):
        """
        :param key_spans: list of containing spans
        :param value_spans: list of contained spans
        key_spans are larger spans which contain value spans without overlap
        example of key, value spans could be (sentence, word) , (doc, word), (doc, sentence)

        creates nodes for each key and value span, and links them
        """
        mapping = tok_seq.match_heirarchical_span_seqs(key_spans, value_spans)

        for key in key_spans:
            key_node = key_node_type(key, subtype=key_node_subtype)

            contained_value_nodes = [value_node_type(val, subtype=value_node_subtype) for val in mapping[key]]
            contained_value_node_ids = existing_graph.add_nodes(contained_value_nodes)

            key_node_id = existing_graph.add_node(key_node)
            edges = [DocumentEdge(key_node_id, cont_node_id, subtype=edge_type)
                     for cont_node_id in contained_value_node_ids]

            existing_graph.add_edges(edges)