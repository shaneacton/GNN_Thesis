import textwrap

from Code.Data.Graph.Nodes.node import Node
from Code.Data.Text.Tokenisation.token_span import TokenSpan


class SpanNode(Node):

    def __init__(self, token_span: TokenSpan):
        super().__init__()
        self.token_span = token_span

    def get_node_viz_text(self):
        text = self.token_span.spacy_span.text + "\n" + repr(self.token_span.token_span)
        return "\n".join(textwrap.wrap(text, 16))

    def __eq__(self, other):
        return self.token_span == other.token_span

    def __hash__(self):
        return hash(self.token_span)