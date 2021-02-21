import os
import textwrap

try:
    import graphviz
except:
    print("python-graphviz not installed. cannot visualise graphs")

from Code.HDE.Graph.graph import HDEGraph
from Code.HDE.Graph.node import HDENode
from Code.constants import CANDIDATE, DOCUMENT, ENTITY


def get_node_colour(node: HDENode):
    colours = {CANDIDATE: "orange", DOCUMENT:  "darkgreen", ENTITY: "cadetblue1"}
    return colours[node.type]


def render_graph(graph: HDEGraph, graph_name="temp", graph_folder="."):

    dot = graphviz.Digraph(comment='The Round Table')
    dot.graph_attr.update({'rankdir': 'LR'})

    name = lambda i: "Node(" + repr(i) + ")"
    ignored_nodes = set()
    for i, node in enumerate(graph.ordered_nodes):
        node: HDENode = node
        if node.type == ENTITY:
            if node.doc_id > 0 or node.ent_token_spen[0] > 20:
                ignored_nodes.add(name(i))
                continue

        if node.type == DOCUMENT:
            if node.doc_id > 1:
                ignored_nodes.add(name(i))
                continue

        if node.type == CANDIDATE:
            if node.candidate_id > 1:
                ignored_nodes.add(name(i))
                continue
        text = textwrap.fill(node.text, 40)
        dot.node(name(i), text, fillcolor=get_node_colour(node), style="filled")

    for edge in graph.ordered_edges:
        # if edge.type() == ENTITY:
        #     continue
        if name(edge.from_id) in ignored_nodes or name(edge.to_id) in ignored_nodes:
            continue
        dot.edge(name(edge.from_id), name(edge.to_id), label=edge.type())

    path = os.path.join('/home/shane/Documents/Thesis/Viz/', graph_folder, graph_name)
    dot.render(path, view=True, format="png")