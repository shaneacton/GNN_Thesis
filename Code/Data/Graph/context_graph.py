import os
from typing import List, Set, Dict

import graphviz

from Code.Config import GraphConstructionConfig
from Code.Data.Graph.Edges.edge_relation import EdgeRelation
from Code.Data.Graph.Nodes.node import Node
from Code.Data.Graph.Nodes.span_node import SpanNode
from Code.Data.Text.Tokenisation.token_span import TokenSpan
from Code.Data.Text.Tokenisation.token_span_hierarchy import TokenSpanHierarchy
from Code.Data.Text.data_sample import DataSample
from Code.Config import graph_construction_config as construction


class ContextGraph:

    """
    Directed graph constructed from a tokensequence and construction config
    """

    def __init__(self, data_sample, span_hierarchy, gcc: GraphConstructionConfig):
        self.gcc = gcc  # the construction config which was used to create this context graph
        self.ordered_nodes: List[Node] = []
        self.node_id_map: Dict[Node, int] = {}  # shortcut to get node id from node ref
        self.span_nodes: Dict[TokenSpan, int] = {}  # maps text segs (eg entities/sentences) to nodes
        self.typed_nodes: Dict[type, Set[int]] = {}  # maps node type to all nodes of that type
        self.query_nodes: Set[int] = set() # collection of nodes which have source=query

        self.unique_edges: Set[EdgeRelation] = set()
        self.ordered_edges: List[EdgeRelation] = []
        self.constructs: List[type] = []  # record of the contruction process used

        self.data_sample: DataSample = data_sample
        self.span_hierarchy: TokenSpanHierarchy = span_hierarchy
        self._query_span_hierarchy: TokenSpanHierarchy = None


    @property
    def query_token_sequence(self):
        # todo multiple questions?
        return self.data_sample.questions[0].token_sequence

    @property
    def query_span_hierarchy(self):
        if not self._query_span_hierarchy:
            self._query_span_hierarchy = TokenSpanHierarchy(self.query_token_sequence)
        return self._query_span_hierarchy

    def get_nodes_of_type(self, type):
        return [self.ordered_nodes[id] for id in self.typed_nodes[type]]

    def get_context_node_ids_at_level(self, level: str):
        spans = self.span_hierarchy[level]
        return [self.span_nodes[span] for span in spans]

    def add_nodes(self, entity_nodes: List[Node]):
        ids = []
        for node in entity_nodes:
            ids.append(self.add_node(node))
        return ids

    def add_node(self, node: Node):
        if node in self.node_id_map.keys():
            return self.node_id_map[node]
        else:
            id = len(self.ordered_nodes)
            self.ordered_nodes.append(node)
            self.node_id_map[node] = id

            if type(node) not in self.typed_nodes.keys():
                self.typed_nodes[type(node)] = set()
            self.typed_nodes[type(node)].add(id)

            if isinstance(node, SpanNode):
                self.span_nodes[node.token_span] = id

            if node.source == construction.QUERY:
                self.query_nodes.add(id)

            return id

    def add_edges(self, edges):
        [self.add_edge(edge) for edge in edges]

    def add_edge(self, edge):
        if edge not in self.unique_edges:
            self.unique_edges.add(edge)
            self.ordered_edges.append(edge)