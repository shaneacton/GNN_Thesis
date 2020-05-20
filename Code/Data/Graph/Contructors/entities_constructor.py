from typing import Union, List, Dict, Tuple

from Code.Data.Graph.Contructors.graph_constructor import GraphConstructor, IncompatibleGraphContructionOrder
from Code.Data.Graph.Edges.same_entity_edge import SameEntityEdge
from Code.Data.Graph.Nodes.entity_node import EntityNode
from Code.Data.Graph.context_graph import ContextGraph


class EntitiesConstructor(GraphConstructor):

    """
    creates a node for each direct mention of an entity,
    meaning one unique entity may have multiple entity nodes.
    links the comentions of each entity
    """

    def append(self, existing_graph, data_sample) -> ContextGraph:
        if existing_graph and len(existing_graph.constructs) != 0:
            raise IncompatibleGraphContructionOrder(existing_graph, self, "entities contruction must be bottom of stack")

        graph = ContextGraph()
        entities = data_sample.context.token_sequence.entities
        entity_nodes = [EntityNode(ent) for ent in entities]
        node_ids = graph.add_nodes(entity_nodes)
        entity_clusters: Dict[Tuple[str], Tuple[int]] = {}  # maps a token list to a set of node ids's who represent these tokens
        for i, node in enumerate(entity_nodes):
            """finds all duplicate entities for linking"""
            id = node_ids[i]
            toks = tuple(node.token_span.tokens)
            if toks not in entity_clusters.keys():
                entity_clusters[toks] = []
            entity_clusters[toks] += [id]

        edge_ids = []  # list of (from, to) node ids
        for unique_toks in entity_clusters.keys():
            """gets edge info for links between duplicate entities"""
            ids = entity_clusters[unique_toks]
            if len(ids) == 1:
                continue
            edge_ids.extend(self.get_edge_ids_connecting_node_ids(ids))

        edges = [SameEntityEdge(edge_id[0], edge_id[1], is_coref=False) for edge_id in edge_ids]
        graph.add_edges(edges)

        graph.constructs.append(type(self))

        return graph

    def get_edge_ids_connecting_node_ids(self, ids):
        edge_ids = []
        for id_1 in ids:
            for id_2 in ids:
                if id_1 == id_2:
                    continue
                edge_ids.append((id_1, id_2))
        return edge_ids

