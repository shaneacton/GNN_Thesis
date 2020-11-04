from abc import ABC, abstractmethod

from transformers import TokenSpan

import Code.constants
from Code.Data.Graph.Nodes.node import Node


class SpanNode(Node, ABC):

    def __init__(self, span: TokenSpan, source=Code.constants.CONTEXT, subtype=None):
        self.token_span = span
        super().__init__(subtype=subtype, source=source)

    def get_node_viz_text(self):
        return "span: " + repr(self.token_span)

    def __eq__(self, other):
        return self.token_span == other.token_span and type(self) == type(other)

    def __hash__(self):
        return hash(self.token_span) + 11 * hash(type(self))

    def __repr__(self):
        return super(SpanNode, self).__repr__() + " - '" + repr(self.token_span) + "'"