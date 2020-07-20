from typing import Dict, Tuple

from Code.Config import config, configuration
from Code.Data.Graph.Contructors.graph_constructor import GraphConstructor
from Code.Data.Graph.Edges.same_entity_edge import SameEntityEdge
from Code.Data.Graph.Nodes.entity_node import EntityNode
from Code.Data.Graph.Nodes.unique_entity_node import UniqueEntityNode
from Code.Data.Graph.context_graph import ContextGraph


class EntitiesConstructor(GraphConstructor):

    """
    creates a node for each direct mention of an entity,
    meaning one unique entity may have multiple entity nodes.
    links the comentions of each entity
    """

    def _append(self, existing_graph: ContextGraph) -> ContextGraph:
        entity_nodes = self.get_entity_nodes(existing_graph)
        node_ids = existing_graph.add_nodes(entity_nodes)

        entity_clusters: Dict[Tuple[str], Tuple[int]] = self.get_entity_clusters(entity_nodes, node_ids)
        edges = self.get_same_edges(entity_clusters)
        if config.has_keyword(configuration.UNIQUE_ENTITY):
            self.add_unique_entity_nodes(existing_graph, entity_clusters)

        existing_graph.add_edges(edges)

        self.add_construct(existing_graph)
        return existing_graph

    @staticmethod
    def get_edge_ids_connecting_node_ids(ids):
        edge_ids = []
        for id_1 in ids:
            for id_2 in ids:
                if id_1 == id_2:
                    continue
                edge_ids.append((id_1, id_2))
        return edge_ids

    @staticmethod
    def get_entity_clusters(entity_nodes, node_ids):
        # maps a raw token list to a set of node ids's who represent these tokens
        entity_clusters: Dict[Tuple[str], Tuple[int]] = {}
        for i, node in enumerate(entity_nodes):
            """finds all duplicate entities for linking"""
            id = node_ids[i]
            toks = tuple(node.token_span.tokens)
            if toks not in entity_clusters.keys():
                entity_clusters[toks] = []
            entity_clusters[toks] += [id]
        return entity_clusters

    def get_same_edges(self, entity_clusters):
        edge_ids = []  # list of (from, to) node ids
        for unique_toks in entity_clusters.keys():
            """gets edge info for links between duplicate entities"""
            ids = entity_clusters[unique_toks]
            if len(ids) == 1:
                continue
            edge_ids.extend(self.get_edge_ids_connecting_node_ids(ids))

        same_edges = [SameEntityEdge(edge_id[0], edge_id[1], to_coref=False) for edge_id in edge_ids]
        return same_edges

    @staticmethod
    def get_entity_nodes(existing_graph):
        entities = existing_graph.span_hierarchy.entities
        entity_nodes = [EntityNode(ent) for ent in entities]
        return entity_nodes

    @staticmethod
    def add_unique_entity_nodes(existing_graph, entity_clusters: Dict[Tuple[str], Tuple[int]]):
        """
        adds the unique entity nodes as well as connecting them to the entity mention nodes
        """

        for unique_toks in entity_clusters.keys():
            if len(entity_clusters[unique_toks]) == 1:
                # this entity only has 1 mention. no need for a unique node
                continue

            mention_node_ids = entity_clusters[unique_toks]
            entity_name: str = existing_graph.ordered_nodes[mention_node_ids[0]].get_node_viz_text()
            entity_name = entity_name.split('(')[0]
            unique_node = UniqueEntityNode(mention_node_ids, entity_name)
            unique_node_id = existing_graph.add_node(unique_node)
            edges = [SameEntityEdge(ment_id, unique_node_id, False, to_unique_node=True) for ment_id in mention_node_ids]
            existing_graph.add_edges(edges)