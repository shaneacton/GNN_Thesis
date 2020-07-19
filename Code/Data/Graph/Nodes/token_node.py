from Code.Data.Graph.Nodes.span_node import SpanNode
from Code.Data.Text.Tokenisation.token_span import TokenSpan


class TokenNode(SpanNode):

    def __init__(self, token_span: TokenSpan, subtype=""):
        super().__init__(token_span)
