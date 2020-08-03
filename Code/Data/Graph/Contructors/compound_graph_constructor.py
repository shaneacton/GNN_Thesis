from typing import List

from Code.Data.Graph.Contructors.graph_constructor import GraphConstructor
from Code.Data.Graph.context_graph import ContextGraph


class CompoundGraphConstructor(GraphConstructor):

    """
    passes a graph through multiple constructors in order
    """

    def __init__(self, constructors: List[type]):
        self.constructors = constructors

    @property
    def type(self):
        return ",".join([repr(const) for const in self.constructors])

    def _append(self, existing_graph: ContextGraph) -> ContextGraph:
        for const_type in self.constructors:
            constructor: GraphConstructor = const_type()
            existing_graph = constructor._append(existing_graph)
            if not existing_graph:
                raise Exception()

        return existing_graph



