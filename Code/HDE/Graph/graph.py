from __future__ import annotations

from typing import List, TYPE_CHECKING, Dict, Generator, Tuple, Set

import numpy as np
import torch
from torch_geometric.utils import remove_self_loops, add_self_loops

from Code.Training import dev
from Code.constants import ENTITY, DOCUMENT, CANDIDATE, GLOBAL, SELF_LOOP, SENTENCE
from Config.config import get_config

if TYPE_CHECKING:
    from Code.HDE.Graph.edge import HDEEdge
    from Code.HDE.Graph.node import HDENode
    from Code.Training.wikipoint import Wikipoint


class HDEGraph:

    def __init__(self, example):
        self.example: Wikipoint = example
        self.ordered_nodes: List[HDENode] = []
        self.entity_nodes: List[int] = []
        self.entity_text_to_nodes: Dict[str, List[int]] = {}
        self.doc_nodes: List[int] = []
        self.candidate_nodes: List[int] = []
        if get_config().use_sentence_nodes:
            self.sentence_inclusion_bools: List[bool] = []
            self.sentence_nodes: List[int] = []

        self.ordered_edges: List[HDEEdge] = []
        self.unique_edges: Set[Tuple[int]] = set()  # set of (t, f) ids, which are always sorted, ie: t<f
        self.unique_edge_types: Set[str] = set()

    def edge_index(self, type=None, direction=None) -> torch.LongTensor:
        """
            if type arg is given, only the edges of that type are returned
            returning edges are automatically added. all evens are outgoing, odds are incoming

            if direction = forwards or backwards, then this will be a unidirectional edge_index,
            with half the edges of the full edge index.

            self loops are added by pytorch geometric
        """
        # if type is None:
        #     raise Exception("weh")
        froms = []
        tos = []

        for e in self.ordered_edges:  # adds both directions
            # print("checking for type:", type, "found:", e.type())
            if type is not None:
                if e.type() != type:
                    continue

            if direction is None:  # both directions
                froms += [e.from_id, e.to_id]
                tos += [e.to_id, e.from_id]
            elif direction == "forward":
                froms += [e.from_id]
                tos += [e.to_id]
            elif direction == "reverse":
                froms += [e.to_id]
                tos += [e.from_id]
            else:
                raise Exception("unreckognised direction type: " + repr(direction) + " must be {forward, reverse}")

        edge_index = torch.tensor([froms, tos]).to(dev()).long()

        if get_config().add_self_loops:
            num_nodes = len(self.ordered_nodes)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        return edge_index

    def get_mask(self):
        # all connections masked. then connections are unmasked
        mask = [[True] * len(self.ordered_nodes) ] * len(self.ordered_nodes)
        for e in self.ordered_edges:  # adds both directions
            mask[e.from_id][e.to_id] = False
            mask[e.to_id][e.from_id] = False
        for i in range(len(self.ordered_nodes)):  # allow self edges
            mask[i][i] = False
        return torch.tensor(mask, dtype=torch.bool).to(dev())

    def ordered_unique_edge_types(self, include_global=False):
        types = sorted(list(self.unique_edge_types))
        types += [SELF_LOOP]
        if include_global:
            types.append(GLOBAL)
        return types

    def edge_types(self):
        types = self.ordered_unique_edge_types()
        edge_type_map = {t: i for i, t in enumerate(types)}
        # print("edge type map:", edge_type_map)
        type_ids = []
        for edge in self.ordered_edges:
            type_id = edge_type_map[edge.type()]
            type_ids.append(type_id)

            if get_config().bidirectional_edge_types:
                reverse_id = edge_type_map[edge.type(reverse=True)]
                type_ids.append(reverse_id)
            else:  # unidirectional - add the same type id twice
                type_ids.append(type_id)

        if get_config().add_self_loops:
            """
                self loops will be automatically appended to the back of the edge index. 
                We must account for those in the type vec. There is 1 self loop per node
            """
            type_ids += [edge_type_map[SELF_LOOP]] * len(self.ordered_nodes)

        return torch.tensor(type_ids).to(dev()).long()

    def get_edge_id_matrix(self):
        """
            here 0 means unconnected. 1 means self edge

            returns: A Lq*Lk matrix of type ids
        """
        types = self.ordered_unique_edge_types()
        edge_type_map = {t: i+2 for i, t in enumerate(types)}
        # print("edge type map:", edge_type_map)
        num_nodes = len(self.ordered_nodes)
        matrix = np.identity(num_nodes)  # all unconnected except for self edges
        for edge in self.ordered_edges:
            type_id = edge_type_map[edge.type()]
            matrix[edge.from_id, edge.to_id] = type_id

            if get_config().bidirectional_edge_types:
                reverse_id = edge_type_map[edge.type(reverse=True)]
                matrix[edge.from_id, edge.to_id] = reverse_id
            else:  # unidirectional - add the same type id twice
                matrix[edge.from_id, edge.to_id] = type_id

        # print("matrix:", matrix)
        return torch.tensor(matrix).to(dev()).long()

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
        if node.type == SENTENCE:
            self.sentence_nodes.append(next_id)

        self.ordered_nodes.append(node)
        return next_id

    def has_edge(self, edge):
        return self.has_connection(edge.to_id, edge.from_id)

    def has_connection(self, to_id, from_id):
        key = tuple(sorted([to_id, from_id]))
        return key in self.unique_edges

    def safe_add_edge(self, edge):
        if not self.has_edge(edge):
            self.add_edge(edge)
            return True
        return False

    def add_edge(self, edge: HDEEdge, safe_mode=False):
        if self.has_edge(edge):
            if safe_mode:
                return
            raise Exception("warning, adding  an edge between two nodes which are already connected")

        self.unique_edge_types.add(edge.type())
        if get_config().bidirectional_edge_types:
            # print("adding reverse edge type:", edge.type(reverse=True))
            self.unique_edge_types.add(edge.type(reverse=True))

        self.ordered_edges.append(edge)
        self.unique_edges.add(tuple(sorted([edge.to_id, edge.from_id])))

    def get_doc_nodes(self) -> List[HDENode]:
        """returns the actual nodes, in order"""
        return [self.ordered_nodes[d] for d in self.doc_nodes]

    def get_candidate_nodes(self) -> List[HDENode]:
        return [self.ordered_nodes[c] for c in self.candidate_nodes]

    def get_entity_nodes(self) -> Generator[HDENode]:
        return (self.ordered_nodes[e] for e in self.entity_nodes)

    def get_cross_document_comention_edges(self) -> Set[HDEEdge]:
        edges = set()
        for edge in self.ordered_edges:
            if "candidate" in edge.type() or "sequential" in edge.type():
                continue
            ent0 = self.ordered_nodes[edge.from_id]
            ent1 = self.ordered_nodes[edge.to_id]
            if ent0.doc_id is None or ent1.doc_id is None:
                raise Exception("no doc id on node " + repr(ent0) + " or " + repr(ent1))
            if ent0.doc_id == ent1.doc_id:
                continue
            edges.add(edge)
        return edges

    @property
    def num_nodes(self):
        return len(self.ordered_nodes)

    @property
    def num_edges(self):
        return len(self.ordered_edges)

    def get_edge_density(self):
        return self.num_edges / (self.num_nodes * (self.num_nodes-1))

    def get_cross_doc_ratio(self):
        return len(self.get_cross_document_comention_edges()) / len(self.doc_nodes)

