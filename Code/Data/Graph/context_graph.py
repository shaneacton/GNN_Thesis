import copy
import os
from typing import List, Set, Dict

import torch

from Code.Data.Graph.Edges.edge_relation import EdgeRelation
from Code.Data.Graph.Nodes.node import Node

from torch_geometric.data import Data

from Code.Data.Graph.Nodes.span_node import SpanNode
from Code.Data.Text.Tokenisation.token_span import TokenSpan

import graphviz

from Code.Training import device
from Datasets.Batching.batch import Batch


class ContextGraph:

    """
    Directed graph
    """

    def __init__(self):
        self.ordered_nodes: List[Node] = []
        self.node_id_map: Dict[Node, int] = {}  # shortcut to get node id from node ref
        self.span_nodes: Dict[TokenSpan, int] = {}  # maps text segs (eg entities/sentences) to nodes
        self.typed_nodes: Dict[type,Set[int]] = {}  # maps node type to all nodes of that type

        self.unique_edges: Set[EdgeRelation] = set()
        self.ordered_edges: List[EdgeRelation] = []
        self.constructs : List[type] = []  # record of the contruction process used

        self.label: torch.Tensor = None
        self.query: torch.Tensor = None

    @property
    def edge_index(self):
        """
        converts edges into connection info for pytorch geometric
        """
        index = [[], []]  # [[from_ids],[to_ids]]
        for edge in self.ordered_edges:
            for from_to in range(2):
                index[from_to].append(edge[from_to])
                if not edge.directed:  # adds returning direction
                    index[from_to].append(edge[1-from_to])
        return index

    @property
    def edge_types(self):
        edge_types = []
        for edge in self.ordered_edges:
            edge_types.append(edge.get_type_tensor())
            if not edge.directed:  # adds returning directions type
                edge_types.append(edge.get_type_tensor())
        return edge_types

    @property
    def data(self):
        """
        converts this graph into data ready to be fed into pytorch geometric
        """
        states_dict: Dict[str, List[torch.Tensor]] = {}
        for node in self.ordered_nodes:  # in order traversal
            states = node.states_tensors
            for state_name in states.keys():
                if not state_name in states_dict:
                    states_dict[state_name] = []
                states_dict[state_name].append(states[state_name])

        for state_name in states_dict.keys():
            states_dict[state_name] = self.pad_and_combine(states_dict[state_name])

        edge_types = torch.stack(self.edge_types, dim=0)
        edge_index = torch.tensor(self.edge_index).to(device)

        # print("states:", {name:states_dict[name].size() for name in states_dict.keys()}, "label:",self.label)
        data_point = Data(edge_index=edge_index, edge_types=edge_types, label=self.label,
                          query=self.query, **states_dict)
        data_point.num_nodes = len(self.ordered_nodes)
        return data_point

    @staticmethod
    def pad_and_combine(vecs: List[torch.Tensor], pad_dim=-1):
        """
        pads dim to equal size, concats along batch dim
        :param pad_dim: the padding dim which contains differing lengths
        :return: batched seqs
        """
        pad_dim = pad_dim if pad_dim != -1 else vecs[0].dim() -1
        longest_seq = max([vec.size(pad_dim) for vec in vecs])

        sizes = list(vecs[0].size())
        sizes = [sizes[i] for i in range(len(sizes)) if i != pad_dim]  # the hetro sizes which aren't the pad dim

        def zeros(curr_len):
            zero_sizes = copy.deepcopy(sizes)
            zero_sizes.insert(pad_dim, longest_seq-curr_len)
            return torch.zeros(*zero_sizes).to(device).long()

        pad = lambda vec: torch.cat([vec, zeros(vec.size(pad_dim))], dim=pad_dim) \
            if vec.size(pad_dim) < longest_seq else vec

        vecs = [pad(vec) for vec in vecs]
        return torch.cat(vecs, dim=0)

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
        if edge not in self.unique_edges:
            self.unique_edges.add(edge)
            self.ordered_edges.append(edge)

    def render_graph(self, graph_name, graph_folder):
        dot = graphviz.Digraph(comment='The Round Table')
        dot.graph_attr.update({'rankdir': 'LR'})

        name = lambda i: "Node(" + repr(i) +")"
        for i, node in enumerate(self.ordered_nodes):
            dot.node(name(i), node.get_node_viz_text())
        for edge in self.unique_edges:
            dot.edge(name(edge[0]), name(edge[1]), label=edge.get_label())

        path = os.path.join('/home/shane/Documents/Thesis/Viz/', graph_folder, graph_name)
        dot.render(path, view=False, format="png")

    def set_label(self, label: torch.Tensor):
        self.label = label

    def set_query(self, query: torch.Tensor):
        self.query = query