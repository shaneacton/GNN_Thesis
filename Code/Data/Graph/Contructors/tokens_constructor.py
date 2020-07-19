from typing import Union

from Code.Data.Graph.Contructors.graph_constructor import GraphConstructor
from Code.Data.Graph.Nodes.token_node import TokenNode
from Code.Data.Graph.context_graph import ContextGraph
from Code.Data.Text.Tokenisation import TokenSpanHierarchy
from Code.Data.Text.data_sample import DataSample


class TokensConstructor(GraphConstructor):
    def append(self, existing_graph: Union[None, ContextGraph], data_sample: DataSample,
               context_span_hierarchy: TokenSpanHierarchy) -> ContextGraph:
        for tok in context_span_hierarchy.tokens:
            node = TokenNode(tok)
            existing_graph.add_node(node)

        return existing_graph