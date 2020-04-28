from Code.GNN_Playground.Data.Answers.answers import Answers
from Code.GNN_Playground.Data.text import Text


class Question(Text):

    def __init__(self, text, answers: Answers = None):
        super().__init__(text)
        self.answers = answers

    def get_all_text_pieces(self):
        return [self.text] + self.answers.get_all_text_pieces()

