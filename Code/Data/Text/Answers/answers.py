from typing import List, Union

import torch

from Code.Data.Text.Answers.answer import Answer
from Code.Data.Text.Answers.candidate_answer import CandidateAnswer
from Code.Data.Text.text import Text
from Code.Models import tail_concatinator


class Answers:

    """
        represents a set of correct answers to a given question
        as well as a set of candidate answers
    """

    def __init__(self, answers: Union[List[Answer],Answer], candidates: List[Answer] = None):
        if isinstance(answers, List):
            self.correct_answers = answers
        if isinstance(answers, Answer):
            self.correct_answers = [answers]

        if isinstance(candidates, List) or candidates is None:
            self.answer_candidates: List[Answer] = candidates
        if isinstance(candidates, Answer):
            raise Exception()

    def get_all_text_pieces(self):
        return [a.text for a in self.correct_answers] + \
               [c.text for c in self.answer_candidates] if self.answer_candidates is not None else []

    def get_answer_type(self):
        return type(self.correct_answers[0])

    def get_candidates_embedding(self):
        if self.get_answer_type() == CandidateAnswer:
            return torch.cat([cand.get_embedding(tail_concatinator) for cand in self.answer_candidates], dim=1)
        raise Exception()

    def get_answer_cand_id(self):
        try:
            return self.answer_candidates.index(self.correct_answers[0])
        except:
            print(self.correct_answers[0], ", cands:",self.answer_candidates)
            print(type(self.correct_answers[0]), type(self.answer_candidates[0]))
            print(self.correct_answers[0] == self.answer_candidates[4])
            raise Exception()
