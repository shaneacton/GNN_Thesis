from typing import List, Set, Dict, Union

from transformers import TokenSpan

import Code.constants
from Code.Data.Graph.Edges.edge_relation import EdgeRelation
from Code.Data.Graph.Nodes.node import Node
from Code.Data.Graph.Nodes.span_node import SpanNode
from Code.Data.Text.text_utils import is_batched


class QAGraph:

    """
    Directed graph constructed from a tokensequence and construction config
    """

    def __init__(self, example, gcc):
        from Code.Config import GraphConstructionConfig
        self.gcc: GraphConstructionConfig = gcc  # the construction config which was used to create this context graph

        self.ordered_nodes: List[Node] = []
        self.node_id_map: Dict[Node, int] = {}  # shortcut to get node id from node ref
        self.span_nodes: Dict[TokenSpan, int] = {}  # maps text segs (eg entities/sentences) to nodes
        self.typed_nodes: Dict[type, Set[int]] = {}  # maps node type to all nodes of that type

        self.query_nodes: Set[int] = set()  # collection of nodes which have source=query
        self.candidate_nodes: Set[int] = set()

        self.ordered_edges: List[EdgeRelation] = []
        self.unique_edges: Set[EdgeRelation] = set()

        self.example = example

        if is_batched(example):
            raise Exception("cannot create graph from a batched example")

    @property
    def has_candidates(self):
        return "candidates" in self.example

    def get_nodes_of_type(self, type):
        return [self.ordered_nodes[id] for id in self.typed_nodes[type]]

    def add_nodes(self, nodes: List[Node]):
        ids = []
        for node in nodes:
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

            if node.source == Code.constants.QUERY:
                self.query_nodes.add(id)

            if node.source == Code.constants.CANDIDATE:
                self.candidate_nodes.add(id)

            return id

    def add_edges(self, edges):
        [self.add_edge(edge) for edge in edges]

    def add_edge(self, edge: EdgeRelation):
        if edge.to_id == edge.from_id:
            # no self connections
            return
        if edge not in self.unique_edges:
            self.unique_edges.add(edge)
            self.ordered_edges.append(edge)