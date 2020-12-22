from typing import List, Union

from transformers import PreTrainedTokenizerFast, BatchEncoding, TokenSpan

from Code.Data.Graph.Contructors.construction_utils import connect_sliding_window, \
    connect_query_and_context, add_nodes_from_hierarchy, connect_candidates_to_graph
from Code.Data.Graph.Nodes.candidate_node import CandidateNode
from Code.Data.Graph.Nodes.structure_node import StructureNode
from Code.Data.Graph.Nodes.word_node import WordNode
from Code.Data.Graph.context_graph import QAGraph
from Code.Data.Text.spacy_utils import get_sentence_char_spans, get_noun_char_spans
from Code.Data.Text.span_hierarchy import SpanHierarchy
from Code.Data.Text.text_utils import context, question, is_batched, candidates, context_key
from Code.Training.Utils.initialiser import get_tokenizer
from Code.Play.examples import test_example
from Code.constants import CONTEXT, QUERY, SENTENCE, WORD, TOKEN, NOUN


class QAGraphConstructor:

    """
    passes a graph through multiple constructors in order
    """

    def __init__(self, gcc, tokeniser: PreTrainedTokenizerFast=None):
        self.gcc: GraphConstructionConfig = gcc
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

            break these up into individual examples to construct graphs individually
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
        if self.gcc.max_context_chars != -1:
            """replace the context in the given example"""
            ctx = context(example)
            # print("cutting ctx len", len(ctx))
            example[context_key(example)] = ctx[0:min(len(ctx), self.gcc.max_context_chars)]  #cuts context
            while example[context_key(example)][-1] == ' ':  # remove trailing spaces
                example[context_key(example)] = example[context_key(example)][0: -1]
        # separates the context and query, calculates node spans from {toks, wrds, sents}, calculates containment
        context_hierarchy, query_hierarchy = self.build_hierarchies(example)
        graph = QAGraph(example, self.gcc)
        add_nodes_from_hierarchy(graph, context_hierarchy)
        add_nodes_from_hierarchy(graph, query_hierarchy)

        connect_sliding_window(graph, context_hierarchy)
        connect_query_and_context(graph)

        if candidates(example):
            cands = candidates(example).split(self.tokeniser.sep_token)
            # print("q:", question(example))
            self.add_candidate_nodes(graph, cands)

        if graph.gcc.max_edges != -1 and len(graph.ordered_edges) > graph.gcc.max_edges:
            raise TooManyEdgesException("data sample created too many edeges ("+str(len(graph.ordered_edges))+
                            ") with this gcc (max = "+str(graph.gcc.max_edges)+"). Discard it")
        return graph

    def add_candidate_nodes(self, graph: QAGraph, cands: List[str]):
        """
            create all candidate nodes, and connect them to the existing graph

            candidates are encoded at the end of a <context><query><all candidates> concat, but separated after embedding.
            Thus each candidate need only know its span in the candidates_string encoding
        """
        nodes = []
        start = 0
        for i, cand in enumerate(cands):
            cand_enc = self.tokeniser(cand)
            num_can_toks = len(cand_enc.tokens())
            end = start + num_can_toks - 1
            span = TokenSpan(start, end)
            nodes.append(CandidateNode(span, i))
            start = end
        graph.add_nodes(nodes)
        connect_candidates_to_graph(graph)

    def build_hierarchies(self, single_example):
        context_encoding: BatchEncoding = self.tokeniser(context(single_example))
        question_encoding: BatchEncoding = self.tokeniser(question(single_example))

        context_hierarchy = SpanHierarchy(context(single_example), context_encoding, CONTEXT)
        query_hierarchy = SpanHierarchy(question(single_example), question_encoding, QUERY, encoding_offset=len(context_encoding.tokens()))

        try:
            self.build_hierarchy(context_hierarchy, CONTEXT)
            self.build_hierarchy(query_hierarchy, QUERY)
        except Exception as e:
            print("failed to add context span nodes for ex " + repr(single_example))
            print("num context chars:", len(context(single_example)), "last few chars:",
                  "'" + context(single_example)[-5:-1] + "'")
            raise e

        return context_hierarchy, query_hierarchy

    def build_hierarchy(self, hierarchy, source):
        structure_levels = self.gcc.structure_levels[source]
        for lev in structure_levels:
            if lev == TOKEN:
                hierarchy.add_tokens()
            elif lev == NOUN:
                hierarchy.add_spans_from_chars(get_noun_char_spans, WORD, WordNode)
            elif lev == SENTENCE:
                if source == CONTEXT:
                    hierarchy.add_spans_from_chars(get_sentence_char_spans, SENTENCE, StructureNode, subtype=SENTENCE)
                else:  # query get a special case
                    hierarchy.add_full_query()
            else:
                raise NotImplementedError()

        hierarchy.calculate_encapsulation()


class TooManyEdgesException(Exception):
    pass


if __name__ == "__main__":
    from Code.Config import gcc, GraphConstructionConfig

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


