import os
import textwrap
from typing import Set, Tuple

from Config.config import get_config

try:
    import graphviz
except:
    print("python-graphviz not installed. cannot visualise graphs")

from Code.HDE.Graph.graph import HDEGraph
from Code.HDE.Graph.node import HDENode
from Code.constants import CANDIDATE, DOCUMENT, ENTITY

name_i = lambda i: "Node(" + repr(i) + ")"
# name = lambda node: "Node(" + repr(node.id_in_graph) + ")"
name = lambda node: "Node(" + repr(node.text_hash) + ")"


def get_node_colour(node: HDENode):
    colours = {CANDIDATE: "orange", DOCUMENT:  "darkgreen", ENTITY: "cadetblue1"}
    return colours[node.type]


def add_doc_nodes(dot, graph: HDEGraph, doc_num, added_node_hashes):
    if doc_num >= len(graph.doc_nodes):
        return doc_num
    doc_id = graph.doc_nodes[doc_num]
    doc_node: HDENode = graph.ordered_nodes[doc_id]
    doc_ents = get_docs_ents(graph, doc_node)

    while len(doc_ents) == 0 and doc_num + 1 < len(graph.doc_nodes):
        doc_num += 1
        doc_id = graph.doc_nodes[doc_num]
        doc_node: HDENode = graph.ordered_nodes[doc_id]
        doc_ents = get_docs_ents(graph, doc_node)
    if len(doc_ents) == 0:
        return doc_num

    dot.node(name(doc_node), repr(doc_id), fillcolor=get_node_colour(doc_node), style="filled", shape="box")
    added_node_hashes.add(doc_node.text_hash)
    added_edge_hashes = set()

    for ent in doc_ents:
        text = textwrap.fill(ent.text, 40)
        edge_id = tuple(sorted([doc_id, ent.id_in_graph]))

        if edge_id not in added_edge_hashes:
            dot.edge(name(doc_node), name(ent), label="doc2ent")
            added_edge_hashes.add(edge_id)

        if ent.text_hash in added_node_hashes:
            continue
        dot.node(name(ent), text, fillcolor=get_node_colour(ent), style="filled")
        added_node_hashes.add(ent.text_hash)

        for cand_id in graph.candidate_nodes:
            edge_id = tuple(sorted([cand_id, ent.id_in_graph]))
            if edge_id not in graph.unique_edges:
                continue
            cand_node = graph.ordered_nodes[cand_id]
            dot.node(name(cand_node), textwrap.fill(cand_node.text, 40), fillcolor=get_node_colour(cand_node),
                     style="filled")
            added_node_hashes.add(cand_node.text_hash)
            dot.edge(name(cand_node), name(ent), label="cand2ent")
    return doc_num


def get_docs_ents(graph, doc_node: HDENode) -> Set[HDENode]:
    ents = set()
    ent_texts = set()
    for ent_id in graph.entity_nodes:
        edge_id = tuple(sorted([doc_node.id_in_graph, ent_id]))
        if edge_id in graph.unique_edges:  # this ent is in this document
            ent_node = graph.ordered_nodes[ent_id]
            if ent_node.text in ent_texts:
                continue
            ents.add(ent_node)
            ent_texts.add(ent_node.text)
    return ents


def render_graph2(graph: HDEGraph, graph_name="temp", graph_folder=None, view=True):
    added_node_hashes: Set[int] = set()

    dot = graphviz.Digraph(comment='The Round Table')
    dot.node("Question", graph.example.query, fillcolor="yellow", style="filled")

    dot.graph_attr.update({'rankdir': 'LR'})
    doc_num = 0
    while doc_num < len(graph.doc_nodes):
        doc_num = add_doc_nodes(dot, graph, doc_num, added_node_hashes)
        doc_num += 1

    if graph_folder is None:
        graph_folder = get_config().model_name

    dot = dot.unflatten(stagger=4)
    dot.render(get_file_path(graph_folder, graph_name), view=view, format="png", cleanup=True)


def get_file_path(folder, file_name):
    path = os.path.join('.', folder, file_name)
    return path