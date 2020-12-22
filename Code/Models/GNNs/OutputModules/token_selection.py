from typing import List

from Code.Data.Graph.Nodes.token_node import TokenNode
from Code.Data.Graph.context_graph import QAGraph
from Code.Models.GNNs.OutputModules.node_selection import NodeSelection


class TokenSelection(NodeSelection):

    def get_node_ids_from_graph(self, graph: QAGraph, source=None, **kwargs) -> List[int]:
        """return the node ids of each token"""
        return super().get_typed_node_ids_from_graph(graph, TokenNode, source=source, **kwargs)


