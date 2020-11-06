from abc import ABC
from typing import Any, Union

from transformers import TokenSpan

import Code.constants
from Code.Data.Graph.Nodes.node import Node


class SpanNode(Node, ABC):

    def __init__(self, span: TokenSpan, source=Code.constants.CONTEXT, subtype=None):
        self.token_span = span
        super().__init__(subtype=subtype, source=source)

    @property
    def start(self):
        return self.token_span.start

    @property
    def end(self):
        return self.token_span.end

    def __eq__(self, other):
        return self.token_span == other.token_span and type(self) == type(other)

    def __hash__(self):
        return hash(self.token_span) + 11 * hash(type(self)) + 17 * hash(self.source)

    def __repr__(self):
        return super(SpanNode, self).__repr__() + " - '" + repr(self.token_span) + "': " + self.source

    def __contains__(self, span: Union[TokenSpan, Any]):
        if isinstance(span, SpanNode):
            span = span.token_span
        left = self.token_span.start <= span.start
        right = self.token_span.end >= span.end
        return left and right