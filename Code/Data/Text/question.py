import torch

from Code.Data.Text.Answers.answers import Answers
from Code.Data.Text.text import Text
from Code.Training import device


class Question(Text):

    def __init__(self, text, answers: Answers = None):
        super().__init__(text)
        self.answers: Answers = answers

    def get_all_text_pieces(self):
        return [self.text] + self.answers.get_all_text_pieces()

    def get_answer_type(self):
        return self.answers.get_answer_type()

    def get_candidates_embedding(self):
        emb = self.answers.get_candidates_embedding()
        return emb

    def get_answer_cand_index_vec(self):
        id = self.answers.get_answer_cand_id()
        return torch.tensor([id]).to(device)
