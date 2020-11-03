from typing import List

from transformers import PreTrainedTokenizerFast

from Code.Data.Graph.Contructors.graph_constructor import GraphConstructor
from Code.Data.Graph.context_graph import ContextGraph
from Code.Data.Text.text_utils import context, question


class QAGraphConstructor(GraphConstructor):

    """
    passes a graph through multiple constructors in order
    """

    def __init__(self, gcc, tokeniser: PreTrainedTokenizerFast):
        super().__init__(gcc)
        self.tokeniser: PreTrainedTokenizerFast = tokeniser

    def _append(self, example, existing_graph: ContextGraph, batch_id=0) -> ContextGraph:
        context_encoding = self.tokeniser(context(example))
        question_encoding = self.tokeniser(question(example))



        if existing_graph.gcc.max_edges != -1 and len(existing_graph.ordered_edges) > existing_graph.gcc.max_edges:
            raise Exception("data sample created too many edeges ("+str(len(existing_graph.ordered_edges))+
                            ") with this gcc (max = "+str(existing_graph.gcc.max_edges)+"). Discard it")

        return existing_graph



