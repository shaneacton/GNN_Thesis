from Code.Data.Graph.Nodes.entity_node import EntityNode
from Code.Data.Text.Tokenisation.token_span import TokenSpan

from Code.Config import graph_construction_config as construction


class CandidateNode(EntityNode):

    def __init__(self, entity: TokenSpan):
        super().__init__(entity, source=construction.CANDIDATE, subtype=construction.CANDIDATE)
