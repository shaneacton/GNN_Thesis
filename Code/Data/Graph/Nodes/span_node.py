import textwrap
from abc import ABC

import Code.constants
from Code.Data.Graph.Nodes.node import Node
from Code.Data.Text.Tokenisation.token_span import TokenSpan
from Code.Config import graph_construction_config as construction


class SpanNode(Node, ABC):

    def __init__(self, token_span: TokenSpan, source=Code.constants.CONTEXT, subtype=None):
        self.token_span = token_span
        super().__init__(subtype=subtype, source=source)

    @property
    def level(self):
        return self.token_span.level

    def get_node_viz_text(self):
        text = "QUERY: " if self.source == Code.constants.QUERY else ""
        text += self.token_span.text + "\n" + repr(self.token_span.subtoken_indexes)
        return "\n".join(textwrap.wrap(text, 24))

    def __eq__(self, other):
        return self.token_span == other.token_span and type(self) == type(other)

    def __hash__(self):
        return hash(self.token_span) + 11 * hash(type(self))

    def __repr__(self):
        return super(SpanNode, self).__repr__() + " - '" + repr(self.token_span) + "'"