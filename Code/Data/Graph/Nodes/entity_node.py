from torch import Tensor

from Code.Data.Graph.Nodes.span_node import SpanNode
from Code.Data.Text.Tokenisation.entity_span import EntitySpan
from Code.Data.Text.Tokenisation.token_span import TokenSpan


class EntityNode(SpanNode):

    COREF = "coref"
    CONTEXT_MENTION = "mention"
    QUERY_ENTITY = "query"
    CANDIDATE = "candidate"

    ALL_TYPES = [COREF, CONTEXT_MENTION, QUERY_ENTITY, CANDIDATE]

    def __init__(self, entity: TokenSpan , subtype=None):
        super().__init__(entity, subtype=subtype)

        subtype = EntityNode.COREF if self.is_coref else EntityNode.CONTEXT_MENTION
        self.set_subtype(subtype)

    @property
    def is_coref(self):
        ent: EntitySpan = self.token_span
        return ent.is_coref

    def get_span_summary_vec(self) -> Tensor:
        return self.token_span.tail_concat_embedding