import os
from textwrap import TextWrapper

import graphviz
from transformers import BatchEncoding

from Code.Config import vizconf
from Code.Data.Graph.Contructors.qa_graph_constructor import QAGraphConstructor
from Code.Data.Graph.Nodes.node import Node
from Code.Data.Graph.Nodes.span_node import SpanNode
from Code.Data.Graph.Nodes.structure_node import StructureNode
from Code.Data.Graph.Nodes.token_node import TokenNode
from Code.Data.Graph.Nodes.word_node import WordNode
from Code.Data.Graph.context_graph import QAGraph
from Code.Data.Text.text_utils import context, question, candidates
from Code.Play.text_encoder import TextEncoder
from Code.Test.examples import test_example
from Code.constants import CONTEXT, QUERY, CANDIDATE, SENTENCE, WORD, TOKEN

wrapper = TextWrapper()


def get_node_text(graph, node: SpanNode, encoding, source, text_encoder: TextEncoder):
    prefix = "Query: " if source == QUERY else ""
    if node.get_structure_level() not in [TOKEN, WORD, SENTENCE]:
        # don't add in full text
        return prefix + node.get_structure_level()
    if node.source == CANDIDATE and node.candidate_id >= vizconf.max_candidates:
        return None
    try:
        s_char = encoding.token_to_chars(node.start + 1)[0]
        e_char = encoding.token_to_chars(node.end)[1]
    except Exception as e:
        print(e)
        raise Exception("could not find " + source + " chars for " + repr(node) + " given encoding of len: " + repr(len(encoding['input_ids'])))
    chars = s_char, e_char
    if chars[1] > vizconf.max_context_graph_chars:
        return None
    full_text = context(graph.example) if node.source == CONTEXT else question(graph.example) if node.source == QUERY \
        else text_encoder.get_cands_string(candidates(graph.example))
    extract = full_text[chars[0]: chars[1]]
    text = prefix + extract
    return wrapper.fill(text)


def render_graph(graph: QAGraph, text_encoder: TextEncoder,
                 graph_name="temp", graph_folder="."):

    dot = graphviz.Digraph(comment='The Round Table')
    dot.graph_attr.update({'rankdir': 'LR'})

    context_encoding: BatchEncoding = text_encoder.get_context_encoding(graph.example)
    query_encoding: BatchEncoding = text_encoder.get_question_encoding(graph.example)
    cands_encoding: BatchEncoding = text_encoder.get_candidates_encoding(graph.example)
    name = lambda i: "Node(" + repr(i) + ")"
    ignored_nodes = set()
    for i, node in enumerate(graph.ordered_nodes):
        node: SpanNode = node
        encoding = context_encoding if node.source == CONTEXT else query_encoding if node.source == QUERY else cands_encoding
        node_text = get_node_text(graph, node, encoding, node.source, text_encoder)
        if node_text is None:
            ignored_nodes.add(name(i))
            continue

        dot.node(name(i), node_text, fillcolor=get_node_colour(node), style="filled")

    for edge in graph.unique_edges:
        if name(edge[0]) in ignored_nodes or name(edge[1]) in ignored_nodes:
            continue
        dot.edge(name(edge[0]), name(edge[1]), label=edge.get_label())

    path = os.path.join('/home/shane/Documents/Thesis/Viz/', graph_folder, graph_name)
    dot.render(path, view=True, format="png")


def get_node_colour(node: Node):
    if node.source == QUERY:
        if isinstance(node, StructureNode):
            return "darkgreen"
        return "green"
    if node.source == CANDIDATE:
        return "orange"

    # is context

    if isinstance(node, TokenNode):
        return "cadetblue1"
    if isinstance(node, WordNode):
        return "deepskyblue4"
    if isinstance(node, StructureNode):
        return "deepskyblue1"

    return "darkolivegreen1"


if __name__ == "__main__":
    from Code.Config import gcc

    const = QAGraphConstructor(gcc)
    print(test_example)
    graph = const._create_single_graph_from_data_sample(test_example)
    context_hierarchy, query_hierarchy = const.build_hierarchies(test_example)
    render_graph(graph, context_hierarchy.encoding, query_hierarchy.encoding, "test", ".")
