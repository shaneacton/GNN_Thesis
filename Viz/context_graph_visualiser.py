import os

import graphviz

import Code.constants
from Code.Data.Graph.Nodes.document_structure_node import DocumentStructureNode
from Code.Data.Graph.Nodes.word_node import EntityNode
from Code.Data.Graph.Nodes.node import Node
from Code.Data.Graph.Nodes.token_node import TokenNode
from Code.Data.Graph.context_graph import ContextGraph
from Code.Config import graph_construction_config as construction


def render_graph(graph: ContextGraph, graph_name, graph_folder):
    dot = graphviz.Digraph(comment='The Round Table')
    dot.graph_attr.update({'rankdir': 'LR'})

    name = lambda i: "Node(" + repr(i) + ")"
    for i, node in enumerate(graph.ordered_nodes):
        dot.node(name(i), node.get_node_viz_text(), fillcolor=get_node_colour(node), style="filled")

    for edge in graph.unique_edges:
        dot.edge(name(edge[0]), name(edge[1]), label=edge.get_label())

    path = os.path.join('/home/shane/Documents/Thesis/Viz/', graph_folder, graph_name)
    dot.render(path, view=False, format="png")

def get_node_colour(node: Node):
    if node.source == Code.constants.QUERY:
        if isinstance(node, DocumentStructureNode):
            return "darkgreen"
        return "green"
    if node.source == Code.constants.CANDIDATE:
        return "orange"

    # is context

    if isinstance(node, TokenNode):
        return "cadetblue1"
    if isinstance(node, EntityNode):
        return "deepskyblue4"
    if isinstance(node, DocumentStructureNode):
        return "deepskyblue1"

    return "darkolivegreen1"

