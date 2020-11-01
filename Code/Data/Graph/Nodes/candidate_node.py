import Code.constants
from Code.Data.Graph.Nodes.word_node import EntityNode



class CandidateNode(EntityNode):

    def __init__(self, entity: TokenSpan, candidate_id):
        super().__init__(entity, source=Code.constants.CANDIDATE, subtype=Code.constants.CANDIDATE)
        self.candidate_id = candidate_id
