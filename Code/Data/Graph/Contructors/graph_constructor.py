from abc import ABC, abstractmethod
from typing import Dict

from transformers import BatchEncoding

from Code.Data.Graph.context_graph import ContextGraph


class GraphConstructor(ABC):

    """
    a graph constructor takes in an optional existing graph, as well as a data sample
    and uses heuristic rules to extract nodes from the sample,
    and connect them to the given existing graph
    finally returning the newly appended to graph
    """

    def __init__(self, gcc):
        self.gcc = gcc

    @abstractmethod
    def _append(self, encoding: BatchEncoding, existing_graph: ContextGraph, batch_id=0) -> ContextGraph:
        raise NotImplementedError()

    def add_construct(self, existing_graph: ContextGraph):
        existing_graph.constructs.append(type(self))

    def create_graph_from_data_sample(self, example):
        graph = ContextGraph(example, gcc=self.gcc)
        return self._append(graph)

    def __call__(self, example: Dict):
        return self.create_graph_from_data_sample(example)


class IncompatibleGraphContructionOrder(Exception):

    def __init__(self, graph: ContextGraph, faulty_constructor: GraphConstructor, message=None):
        self.graph = graph
        self.faulty_constructor = faulty_constructor
        self.message = message

    def __str__(self):
        return "cannot stack " + repr(self.faulty_constructor) + " on top of graph with " \
            + repr(self.graph.constructs) + " contructs" + ("\n("+self.message+")" if self.message else "")
