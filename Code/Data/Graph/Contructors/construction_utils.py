from typing import List

from Code.Data.Graph.Edges.candidate_edge import CandidateEdge
from Code.Data.Graph.Edges.query_edge import QueryEdge
from Code.Data.Graph.Edges.structure_edge import StructureEdge
from Code.Data.Graph.Edges.window_edge import WindowEdge
from Code.Data.Graph.context_graph import QAGraph
from Code.Data.Text.span_hierarchy import SpanHierarchy
from Code.Play.initialiser import ATTENTION_WINDOW


def add_nodes_from_hierarchy(graph: QAGraph, hierarchy: SpanHierarchy, connect=True):
    """adds nodes at each level, and adds in the hierarchical connections"""
    for lev in hierarchy.present_levels:
        # print("adding nodes:", hierarchy.levels[lev])
        graph.add_nodes(hierarchy.levels[lev])
    if not connect:
        return
    for from_s in hierarchy.containing_links:
        to_nodes = hierarchy.containing_links[from_s]
        for to in to_nodes:
            edge = get_structure_edge(graph, from_s, to)
            graph.add_edge(edge)


def connect_sliding_window(graph: QAGraph, hierarchy: SpanHierarchy, window_size=ATTENTION_WINDOW):
    for lev in hierarchy.levels:
        nodes = hierarchy.levels[lev]
        for n, node in enumerate(nodes):
            for d in range(1, window_size):
                # todo check. this seems wrong, n is not the from id
                to_id = n + d
                if to_id >= len(nodes):
                    break
                to_node = nodes[to_id]
                graph.add_edge(get_window_edge(graph, node, to_node, d))


def connect_candidates_to_graph(graph: QAGraph):
    c_nodes = [graph.ordered_nodes[q_node] for q_node in graph.candidate_nodes]
    for c_node in c_nodes:
        """connect each candidate node to every other node, including self"""
        for node in graph.ordered_nodes:
            graph.add_edge(get_candidate_edge(graph, c_node, node))


def connect_query_and_context(graph: QAGraph):
    q_nodes = [graph.ordered_nodes[q_node] for q_node in graph.query_nodes]
    for q_node in q_nodes:
        """connect each query node to every other node, including self"""
        for node in graph.ordered_nodes:
            graph.add_edge(get_query_edge(graph, q_node, node))


def get_node_ids(graph, f_node, t_node):
    return graph.node_id_map[f_node], graph.node_id_map[t_node]


def get_candidate_edge(graph, c_node, node):
    f_id, t_id = get_node_ids(graph, c_node, node)
    return CandidateEdge(f_id, t_id, node.get_structure_level(), node.source)


def get_query_edge(graph, q_node, node):
    f_id, t_id = get_node_ids(graph, q_node, node)
    return QueryEdge(f_id, t_id, q_node.get_structure_level(), node.get_structure_level())


def get_window_edge(graph, f_node, t_node, distance):
    f_id, t_id = get_node_ids(graph, f_node, t_node)
    return WindowEdge(f_id, t_id, f_node.get_structure_level(), distance)


def get_structure_edge(graph, f_node, t_node):
    f_id, t_id = get_node_ids(graph, f_node, t_node)
    type = f_node.get_structure_level() + "2" + t_node.get_structure_level()
    return StructureEdge(f_id, t_id, subtype=type)
