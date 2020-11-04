from transformers import PreTrainedTokenizerFast

from Code.Data.Graph.Contructors.graph_constructor import GraphConstructor
from Code.Data.Graph.context_graph import ContextGraph
from Code.Data.Text.spacy_utils import get_sentence_char_spans, get_flat_entity_and_corefs_chars, get_noun_char_spans
from Code.Data.Text.span_hierarchy import SpanHierarchy
from Code.Data.Text.text_utils import context, question
from Code.Play.initialiser import get_tokenizer
from Code.Test.examples import test_example
from Code.constants import CONTEXT, QUERY, SENTENCE, WORD


class QAGraphConstructor(GraphConstructor):

    """
    passes a graph through multiple constructors in order
    """

    def __init__(self, gcc, tokeniser: PreTrainedTokenizerFast=None):
        super().__init__(gcc)
        if not tokeniser:
            tokeniser = get_tokenizer()
        self.tokeniser: PreTrainedTokenizerFast = tokeniser

    def _append(self, example, existing_graph: ContextGraph, batch_id=0) -> ContextGraph:
        context_encoding = self.tokeniser(context(example))
        question_encoding = self.tokeniser(question(example))

        context_hierarchy = SpanHierarchy(context(example), context_encoding, CONTEXT)
        query_hierarchy = SpanHierarchy(question(example), question_encoding, QUERY)

        context_hierarchy.add_tokens()
        context_hierarchy.add_spans_from_chars(get_noun_char_spans, WORD)
        context_hierarchy.add_spans_from_chars(get_sentence_char_spans, SENTENCE)
        context_hierarchy.calculate_encapsulation()
        # ls = context_hierarchy.containing_links
        # print("\n".join([repr(l) for l in ls.items()]))
        # for l in ls:
        #     toks = context_encoding.tokens()[l.start:l.end]
        #     conts = [context_encoding.tokens()[c.start:c.end] for c in ls[l]]
        #     print("big:",toks)
        #     print("smol:", conts)

        query_hierarchy.add_tokens()
        query_hierarchy.add_full_query()
        query_hierarchy.calculate_encapsulation()


        # if existing_graph.gcc.max_edges != -1 and len(existing_graph.ordered_edges) > existing_graph.gcc.max_edges:
        #     raise Exception("data sample created too many edeges ("+str(len(existing_graph.ordered_edges))+
        #                     ") with this gcc (max = "+str(existing_graph.gcc.max_edges)+"). Discard it")
        #
        # return existing_graph


if __name__ == "__main__":
    from Code.Config import gcc
    const = QAGraphConstructor(gcc)
    print(test_example)
    const._append(test_example, None)


