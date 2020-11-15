from typing import List, Union

from transformers import PreTrainedTokenizerFast, BatchEncoding

from Code.Data.Graph.Contructors.construction_utils import get_structure_edge, connect_sliding_window, \
    connect_query_and_context
from Code.Data.Graph.Nodes.structure_node import StructureNode
from Code.Data.Graph.Nodes.word_node import WordNode
from Code.Data.Graph.context_graph import QAGraph
from Code.Data.Text.spacy_utils import get_sentence_char_spans, get_noun_char_spans
from Code.Data.Text.span_hierarchy import SpanHierarchy
from Code.Data.Text.text_utils import context, question, is_batched, question_key, candidates
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

    def __call__(self, example) -> Union[List[QAGraph], QAGraph]:
        if is_batched(example):
            return self._create_graphs_from_batched_data_sample(example)
        return self._create_single_graph_from_data_sample(example)

    def _create_graphs_from_batched_data_sample(self, example) -> List[QAGraph]:
        """
            The torch data loader batches dictionaries by stacking each of the values per key
            {a:1, b:2} + {a:3, b:4} = {a: [1,3], b:[2,4]}

            break these up into individual examples to construct graphs in series
            todo: parallelise
        """

        single_examples = []
        questions = question(example)
        contexts = context(example)
        cands = candidates(example)
        # print("cands:", cands, "ex:", example)
        for i in range(len(questions)):
            example = {"context": contexts[i], 'question': questions[i]}
            example.update({"candidates": cands[i]} if cands else {})
            single_examples.append(example)
        graphs = [self._create_single_graph_from_data_sample(ex) for ex in single_examples]
        return graphs

    def _create_single_graph_from_data_sample(self, example) -> QAGraph:
        if is_batched(example):
            raise Exception("must pass a single, unbatched example. instead got: " + repr(example))

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

    def build_hierarchies(self, single_example):
        context_encoding: BatchEncoding = self.tokeniser(context(single_example))
        # print("num context tokens:", len(context_encoding.tokens()), "ctx:", context_encoding.tokens())
        question_encoding: BatchEncoding = self.tokeniser(question(single_example))

        context_hierarchy = SpanHierarchy(context(single_example), context_encoding, CONTEXT)
        query_hierarchy = SpanHierarchy(question(single_example), question_encoding, QUERY)

        context_hierarchy.add_tokens()
        try:
            context_hierarchy.add_spans_from_chars(get_noun_char_spans, WORD, WordNode)
            context_hierarchy.add_spans_from_chars(get_sentence_char_spans, SENTENCE, StructureNode, subtype=SENTENCE)
        except Exception as e:
            print(e)
            raise Exception("failed to add context span nodes for ex " + repr(single_example) + "\nnum context chars:" + repr(len(context(single_example))))
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
    const._create_single_graph_from_data_sample(test_example)
    context_hierarchy, query_hierarchy = const.build_hierarchies(test_example)

    ls = context_hierarchy.containing_links
    print("\n".join([repr(l) for l in ls.items()]))
    for l in ls:
        toks = context_hierarchy.encoding.tokens()[l.start:l.end]
        conts = [context_hierarchy.encoding.tokens()[c.start:c.end] for c in ls[l]]
        print("big:",toks)
        print("smol:", conts)


