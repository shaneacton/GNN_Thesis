from abc import ABC, abstractmethod
from typing import Union

from Code.Data.Graph.context_graph import ContextGraph
from Code.Data.Text.data_sample import DataSample


class GraphConstructor(ABC):

    """
    a graph constructor takes in an optional existing graph, as well as a data sample
    and uses heuristic rules to extract nodes from the sample,
    and connect them to the given existing graph
    finally returning the newly appended to graph
    """

    @abstractmethod
    def append(self, existing_graph:Union[None, ContextGraph], data_sample: DataSample) -> ContextGraph:
        raise NotImplementedError()

    def create_graph_from_data_sample(self, data_sample: DataSample):
        return self.append(None, data_sample)


class IncompatibleGraphContructionOrder(Exception):

    def __init__(self, graph: ContextGraph, faulty_constructor: GraphConstructor, message=None):
        self.graph = graph
        self.faulty_constructor = faulty_constructor
        self.message = message

    def __str__(self):
        return "cannot stack " + repr(self.faulty_constructor) + " on top of graph with " \
            + repr(self.graph.constructs) + " contructs" + ("\n("+self.message+")" if self.message else "")
