from Code.Data.Graph.Contructors.graph_constructor import GraphConstructor
from Code.Data.Graph.Nodes.token_node import TokenNode
from Code.Data.Graph.context_graph import ContextGraph


class TokensConstructor(GraphConstructor):
    def _append(self, existing_graph: ContextGraph) -> ContextGraph:
        for tok in existing_graph.span_hierarchy.tokens:
            node = TokenNode(tok)
            existing_graph.add_node(node)

        self.add_construct(existing_graph)
        return existing_graph