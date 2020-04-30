from typing import List, Union

from Code.GNN_Playground.Data.Answers.answer import Answer


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
            self.answer_candidates = candidates
        if isinstance(candidates, Answer):
            raise Exception()

    def get_all_text_pieces(self):
        return [a.text for a in self.correct_answers] + \
               [c.text for c in self.answer_candidates] if self.answer_candidates is not None else []

    def get_answer_type(self):
        return type(self.correct_answers[0])
