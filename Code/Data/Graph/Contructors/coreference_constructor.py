from typing import Union, List

from Code.Data.Graph.Contructors.entities_constructor import EntitiesConstructor
from Code.Data.Graph.Contructors.graph_constructor import GraphConstructor, IncompatibleGraphContructionOrder
from Code.Data.Graph.Edges.same_entity_edge import SameEntityEdge
from Code.Data.Graph.Nodes.entity_node import EntityNode
from Code.Data.Graph.context_graph import ContextGraph
from Code.Data.Text.Tokenisation import TokenSpanHierarchy
from Code.Data.Text.data_sample import DataSample


class CoreferenceConstructor(GraphConstructor):

    """
    goes through each entity node in the given graph,
    finds the coreference entites for that node and
    links the new coreference nodes to the existing graph
    """

    def append(self, existing_graph: Union[None, ContextGraph], data_sample: DataSample) -> ContextGraph:
        if not existing_graph or EntitiesConstructor not in existing_graph.constructs:
            raise IncompatibleGraphContructionOrder(existing_graph, self, "Entities must be graphed before coreferences")

        try:
            entity_nodes: List[EntityNode] = existing_graph.get_nodes_of_type(EntityNode)
        except Exception as e:
            raise Exception("Failed to add coref nodes\n\n"+repr(e) + "\n\nexisting nodes:", existing_graph.ordered_nodes, "text ents")

        tok_seq = data_sample.context.token_sequence
        span_hierarchy = TokenSpanHierarchy(tok_seq)
        corefs = span_hierarchy.corefs
        for node in entity_nodes:
            ent = node.token_span
            if ent not in corefs.keys():
                continue  # no corefs for this entity

            coref_nodes = [EntityNode(coref_ent) for coref_ent in corefs[ent]]
            #remove corefs which are already linked by SAME edge
            coref_nodes = [node for node in coref_nodes if node not in existing_graph.node_id_map]
            ids = existing_graph.add_nodes(coref_nodes)
            ent_node_id = existing_graph.node_id_map[node]

            edges = [SameEntityEdge(ent_node_id, coref_id, is_coref=True) for coref_id in ids]
            existing_graph.add_edges(edges)

        existing_graph.constructs.append(type(self))
        return existing_graph
            
