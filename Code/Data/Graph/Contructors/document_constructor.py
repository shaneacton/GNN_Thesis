from typing import Union

from Code.Data.Graph.Contructors.graph_constructor import GraphConstructor
from Code.Data.Graph.context_graph import ContextGraph
from Code.Data.Text.data_sample import DataSample


class DocumentConstructor(GraphConstructor):

    """links up nodes"""

    def append(self, existing_graph: Union[None, ContextGraph], data_sample: DataSample) -> ContextGraph:
        pass