from transformers import PreTrainedTokenizerFast, BatchEncoding

from Code.Data.Graph.Contructors.construction_utils import get_structure_edge, connect_sliding_window, \
    connect_query_and_context
from Code.Data.Graph.Nodes.structure_node import StructureNode
from Code.Data.Graph.Nodes.word_node import WordNode
from Code.Data.Graph.context_graph import QAGraph
from Code.Data.Text.spacy_utils import get_sentence_char_spans, get_noun_char_spans
from Code.Data.Text.span_hierarchy import SpanHierarchy
from Code.Data.Text.text_utils import context, question
from Code.Play.initialiser import get_tokenizer
from Code.Test.examples import test_example
from Code.constants import CONTEXT, QUERY, SENTENCE, WORD


class QAGraphConstructor:

    """
    passes a graph through multiple constructors in order
    """

    def __init__(self, gcc, tokeniser: PreTrainedTokenizerFast=None):
        self.gcc = gcc
        if not tokeniser:
            tokeniser = get_tokenizer()
        self.tokeniser: PreTrainedTokenizerFast = tokeniser

    def __call__(self, example, batch_id=0):
        return self.create_graph_from_data_sample(example, batch_id=batch_id)

    def create_graph_from_data_sample(self, example, batch_id=0) -> QAGraph:
        context_hierarchy, query_hierarchy = self.build_hierarchies(example)
        graph = QAGraph(example, self.gcc)
        self.add_nodes_from_hierarchy(graph, context_hierarchy)
        self.add_nodes_from_hierarchy(graph, query_hierarchy)

        connect_sliding_window(graph, context_hierarchy)
        connect_sliding_window(graph, query_hierarchy)
        connect_query_and_context(graph)

        if graph.gcc.max_edges != -1 and len(graph.ordered_edges) > graph.gcc.max_edges:
            raise Exception("data sample created too many edeges ("+str(len(graph.ordered_edges))+
                            ") with this gcc (max = "+str(graph.gcc.max_edges)+"). Discard it")
        return graph

    def build_hierarchies(self, example):
        context_encoding: BatchEncoding = self.tokeniser(context(example))
        question_encoding: BatchEncoding = self.tokeniser(question(example))

        context_hierarchy = SpanHierarchy(context(example), context_encoding, CONTEXT)
        query_hierarchy = SpanHierarchy(question(example), question_encoding, QUERY)

        context_hierarchy.add_tokens()
        context_hierarchy.add_spans_from_chars(get_noun_char_spans, WORD, WordNode)
        context_hierarchy.add_spans_from_chars(get_sentence_char_spans, SENTENCE, StructureNode, subtype=SENTENCE)
        context_hierarchy.calculate_encapsulation()

        query_hierarchy.add_tokens()
        query_hierarchy.add_full_query()
        query_hierarchy.calculate_encapsulation()
        return context_hierarchy, query_hierarchy

    def add_nodes_from_hierarchy(self, graph: QAGraph, hierarchy: SpanHierarchy, connect=True):
        """adds nodes at each level, and adds in the hierarchical connections"""
        for lev in hierarchy.present_levels:
            graph.add_nodes(hierarchy.levels[lev])
        if not connect:
            return
        for from_s in hierarchy.containing_links:
            to_nodes = hierarchy.containing_links[from_s]
            for to in to_nodes:
                edge = get_structure_edge(graph, from_s, to)
                graph.add_edge(edge)


if __name__ == "__main__":
    from Code.Config import gcc
    const = QAGraphConstructor(gcc)
    print(test_example)
    const.create_graph_from_data_sample(test_example)
    context_hierarchy, query_hierarchy = const.build_hierarchies(test_example)

    ls = context_hierarchy.containing_links
    print("\n".join([repr(l) for l in ls.items()]))
    for l in ls:
        toks = context_hierarchy.encoding.tokens()[l.start:l.end]
        conts = [context_hierarchy.encoding.tokens()[c.start:c.end] for c in ls[l]]
        print("big:",toks)
        print("smol:", conts)


