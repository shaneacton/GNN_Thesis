import textwrap

from Code.Data.Graph.Nodes.span_node import SpanNode
from Code.Data.Text.Tokenisation.token_span import TokenSpan


class QueryNode(SpanNode):
    """
    can be a token from the query token sequence
    or a single node representing the whole query
    """

    def __init__(self, token_span: TokenSpan, subtype):
        super().__init__(token_span, subtype=subtype)

    def get_node_viz_text(self):
        text = "QUERY: " + self.token_span.text
        return "\n".join(textwrap.wrap(text, 16))

    def get_structure_level(self):
        return self.subtype