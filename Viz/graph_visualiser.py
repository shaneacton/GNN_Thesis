import os
import textwrap

from Config.config import get_config

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


def render_graph(graph: HDEGraph, graph_name="temp", graph_folder=None, view=True):

    dot = graphviz.Digraph(comment='The Round Table')
    dot.graph_attr.update({'rankdir': 'LR'})

    name = lambda i: "Node(" + repr(i) + ")"
    ignored_nodes = set()
    num_doc0_ents = 0
    num_doc1_ents = 0
    for i, node in enumerate(graph.ordered_nodes):
        node: HDENode = node
        if node.type == ENTITY:
            if node.doc_id > 1:
                ignored_nodes.add(name(i))
                continue
            if node.doc_id == 0:
                if num_doc0_ents >= 4:
                    ignored_nodes.add(name(i))
                    continue
                else:
                    num_doc0_ents += 1

            elif node.doc_id == 1:
                if num_doc1_ents >= 1:
                    ignored_nodes.add(name(i))
                    continue
                else:
                    num_doc1_ents += 1

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

    if graph_folder is None:
        graph_folder = get_config().model_name
    dot.render(get_file_path(graph_folder, graph_name), view=view, format="png", cleanup=True)


def get_file_path(folder, file_name):
    path = os.path.join('.', folder, file_name)
    return path