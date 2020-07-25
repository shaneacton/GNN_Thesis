from abc import ABC, abstractmethod

from Code.Config import gcc
from Code.Data.Graph.context_graph import ContextGraph
from Code.Data.Text.Tokenisation import TokenSpanHierarchy
from Code.Data.Text.data_sample import DataSample


class GraphConstructor(ABC):

    """
    a graph constructor takes in an optional existing graph, as well as a data sample
    and uses heuristic rules to extract nodes from the sample,
    and connect them to the given existing graph
    finally returning the newly appended to graph
    """

    @abstractmethod
    def _append(self, existing_graph: ContextGraph) -> ContextGraph:
        raise NotImplementedError()

    def add_construct(self, existing_graph: ContextGraph):
        existing_graph.constructs.append(type(self))

    def create_graph_from_data_sample(self, data_sample: DataSample):
        context_span_hierarchy = TokenSpanHierarchy(data_sample.context.token_sequence)
        graph = ContextGraph(data_sample, context_span_hierarchy, gcc=gcc)
        return self._append(graph)


class IncompatibleGraphContructionOrder(Exception):

    def __init__(self, graph: ContextGraph, faulty_constructor: GraphConstructor, message=None):
        self.graph = graph
        self.faulty_constructor = faulty_constructor
        self.message = message

    def __str__(self):
        return "cannot stack " + repr(self.faulty_constructor) + " on top of graph with " \
            + repr(self.graph.constructs) + " contructs" + ("\n("+self.message+")" if self.message else "")
