import Code.constants
from Code.Data.Graph.Nodes.entity_node import EntityNode
from Code.Data.Text.Tokenisation.token_span import TokenSpan

from Code.Config import graph_construction_config as construction


class CandidateNode(EntityNode):

    def __init__(self, entity: TokenSpan, candidate_id):
        super().__init__(entity, source=Code.constants.CANDIDATE, subtype=Code.constants.CANDIDATE)
        self.candidate_id = candidate_id
