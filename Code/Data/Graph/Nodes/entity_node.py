from Code.Config import graph_construction_config as construction
from Code.Data.Graph.Nodes.span_node import SpanNode
from Code.Data.Text.Tokenisation.entity_span import EntitySpan


class EntityNode(SpanNode):

    def __init__(self, entity: EntitySpan, subtype=None):
        if not subtype:
            subtype = entity.get_subtype()
        super().__init__(entity, subtype=subtype)

    @property
    def is_coref(self):
        ent: EntitySpan = self.token_span
        return ent.is_coref

    def get_structure_level(self):
        return construction.WORD
