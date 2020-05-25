from typing import Dict

from torch import Tensor

from Code.Data.Graph.Nodes.span_node import SpanNode
from Code.Data.Text.Tokenisation.entity import Entity
from Code.Data.Text.Tokenisation.token_span import TokenSpan


class EntityNode(SpanNode):

    """
    entry level node which contains only a current state vector
    """


    def __init__(self, entity:TokenSpan):
        super().__init__(entity)
        self.subtype = repr(self.is_coref)

    def get_starting_state(self) -> Tensor:
        return self.token_span.tail_concat_embedding

    @property
    def is_coref(self):
        ent: Entity = self.token_span
        return ent.is_coref

    # def get_node_states(self) -> Dict[str, Tensor]:
    #     return {"current_state": self.current_state}
