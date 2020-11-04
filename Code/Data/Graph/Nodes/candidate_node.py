from transformers import TokenSpan

import Code.constants
from Code.Data.Graph.Nodes.word_node import WordNode


class CandidateNode(WordNode):

    def __init__(self, span: TokenSpan, candidate_id):
        super().__init__(span, source=Code.constants.CANDIDATE, subtype=Code.constants.CANDIDATE)
        self.candidate_id = candidate_id
