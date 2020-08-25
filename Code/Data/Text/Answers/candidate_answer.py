from Code.Data.Text.Answers.answer import Answer
from Code.Models.GNNs.OutputModules.candidate_selection import CandidateSelection


class CandidateAnswer(Answer):

    def __init__(self, text: str):
        super().__init__(text)

    def get_output_model(self):
        return CandidateSelection
