import os
from typing import List, Set, Dict

import torch

from Code.Data.Graph.Edges.edge_relation import EdgeRelation
from Code.Data.Graph.Nodes.node import Node

from torch_geometric.data import Data

from Code.Data.Graph.Nodes.span_node import SpanNode
from Code.Data.Text.Tokenisation.token_span import TokenSpan

import graphviz


class ContextGraph:

    """
    Directed graph
    """

    def __init__(self):
        self.ordered_nodes: List[Node] = []
        self.node_id_map: Dict[Node, int] = {}  # shortcut to get node id from node ref
        self.span_nodes: Dict[TokenSpan, int] = {}  # maps text segs (eg entities/sentences) to nodes
        self.typed_nodes: Dict[type,Set[int]] = {}  # maps node type to all nodes of that type

        self.edges: Set[EdgeRelation] = set()
        self.constructs : List[type] = []  # record of the contruction process used

        self.label: torch.Tensor = None

    @property
    def edge_index(self):
        """
        converts edges into connection info for pytorch geometric
        """
        index = [[], []]
        for edge in self.edges:
            for i in range(2):
                index[i].append(edge[i])
                if not edge.directed:  # adds returning direction
                    index[i].append(edge[1-i])
        return index

    @property
    def data(self):
        """
        converts this graph into data ready to be fed into pytorch geometric
        """
        states_dict = {}
        for node in self.ordered_nodes:  # in order traversal
            for state_name in node.states.keys():
                if not state_name in states_dict:
                    states_dict[state_name] = []
                states_dict[state_name].append(node.states[state_name])

        for state_name in states_dict.keys():
            states_dict[state_name] = torch.stack(states_dict[state_name], dim=0)
        print("states:", {name:states_dict[name].size() for name in states_dict.keys()}, "label:",self.label)
        return Data(edge_index=self.edge_index, label=self.label, **states_dict)

    def get_nodes_of_type(self, type):
        return [self.ordered_nodes[id] for id in self.typed_nodes[type]]

    def add_nodes(self, entity_nodes: List[Node]):
        ids = []
        for node in entity_nodes:
            ids.append(self.add_node(node))
        return ids

    def add_node(self, node):
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
            return id

    def add_edges(self, edges):
        [self.add_edge(edge) for edge in edges]

    def add_edge(self, edge):
        self.edges.add(edge)

    def render_graph(self, graph_name, graph_folder):
        dot = graphviz.Digraph(comment='The Round Table')
        dot.graph_attr.update({'rankdir': 'LR'})

        name = lambda i: "Node(" + repr(i) +")"
        for i, node in enumerate(self.ordered_nodes):
            dot.node(name(i), node.get_node_viz_text())
        for edge in self.edges:
            dot.edge(name(edge[0]), name(edge[1]), label=edge.get_label())

        path = os.path.join('/home/shane/Documents/Thesis/Viz/', graph_folder, graph_name)
        dot.render(path, view=False, format="png")

    def set_label(self, label: torch.Tensor):
        self.label = label


if __name__ == "__main__":
    x = torch.tensor([[2, 1], [5, 6], [3, 7], [12, 0]], dtype=torch.float)
    print(x.size())