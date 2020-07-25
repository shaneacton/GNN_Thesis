from Code.Data.Graph.Nodes.span_node import SpanNode
from Code.Data.Text.Tokenisation.token_span import TokenSpan
from Code.Config import graph_construction_config as construction


class TokenNode(SpanNode):

    def __init__(self, token_span: TokenSpan, subtype=""):
        super().__init__(token_span)

    def get_structure_level(self):
        return construction.TOKEN
