from __future__ import annotations

from typing import List, TYPE_CHECKING, Dict, Generator, Tuple, Set

import torch

from Code.Training import device
from Code.constants import ENTITY, DOCUMENT, CANDIDATE

if TYPE_CHECKING:
    from Code.HDE.Graph.edge import HDEEdge
    from Code.HDE.Graph.node import HDENode


class HDEGraph:

    def __init__(self, example):
        self.example = example
        self.ordered_nodes: List[HDENode] = []
        self.entity_nodes: List[int] = []
        self.entity_text_to_nodes: Dict[str, List[int]] = {}
        self.doc_nodes: List[int] = []
        self.candidate_nodes: List[int] = []

        self.ordered_edges: List[HDEEdge] = []
        self.unique_edges: Set[Tuple[int]] = set()  # set of (t, f) ids, which are always sorted, ie: t<f
        self.unique_edge_types: Set[str] = set()

    def edge_index(self, type=None) -> torch.LongTensor:
        """if type arg is given, only the edges of that type are returned"""
        froms = []
        tos = []

        for e in self.ordered_edges:  # adds both directions
            if type is not None:
                if e.type() != type:
                    continue
            froms += [e.from_id, e.to_id]
            tos += [e.to_id, e.from_id]
        return torch.tensor([froms, tos]).to(device).long()

    def edge_types(self):
        types = sorted(list(set([edge.type() for edge in self.ordered_edges])))
        edge_type_map = {t: i for i, t in enumerate(types)}
        type_ids = []
        for edge in self.ordered_edges:
            type_id = edge_type_map[edge.type()]
            type_ids.append(type_id)
        return torch.tensor(type_ids).to(device).long()

    def add_node(self, node: HDENode) -> int:
        next_id = len(self.ordered_nodes)
        node.id_in_graph = next_id
        if node.type == ENTITY:
            node.ent_id = len(self.entity_nodes)
            self.entity_nodes.append(next_id)
            if not node.text in self.entity_text_to_nodes:
                self.entity_text_to_nodes[node.text] = []
            self.entity_text_to_nodes[node.text].append(next_id)

        if node.type == DOCUMENT:
            self.doc_nodes.append(next_id)
        if node.type == CANDIDATE:
            self.candidate_nodes.append(next_id)

        self.ordered_nodes.append(node)
        return next_id

    def has_edge(self, edge):
        key = tuple(sorted([edge.to_id, edge.from_id]))
        return key in self.unique_edges

    def has_connection(self, to_id, from_id):
        key = tuple(sorted([to_id, from_id]))
        return key in self.unique_edges

    def safe_add_edge(self, edge):
        if not self.has_edge(edge):
            self.add_edge(edge)
            return True
        return False

    def add_edge(self, edge: HDEEdge):
        if self.has_edge(edge):
            print("warning, adding  an edge between two nodes which are already connected")
        self.unique_edge_types.add(edge.type())
        self.ordered_edges.append(edge)
        self.unique_edges.add(tuple(sorted([edge.to_id, edge.from_id])))

    def get_doc_nodes(self) -> List[HDENode]:
        """returns the actual nodes, in order"""
        return [self.ordered_nodes[d] for d in self.doc_nodes]

    def get_candidate_nodes(self) -> List[HDENode]:
        return [self.ordered_nodes[c] for c in self.candidate_nodes]

    def get_entity_nodes(self) -> Generator[HDENode]:
        return (self.ordered_nodes[e] for e in self.entity_nodes)
