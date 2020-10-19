from typing import List, Union

from Code.Data.Text.Answers.answer import Answer


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
        return [a.raw_text for a in self.correct_answers] + \
               [c.raw_text for c in self.answer_candidates] if self.answer_candidates is not None else []

    def get_answer_type(self):
        return type(self.correct_answers[0])

    def get_answer_cand_id(self):
        try:
            return self.answer_candidates.index(self.correct_answers[0])
        except:
            print(self.correct_answers[0], ", cands:",self.answer_candidates)
            print(type(self.correct_answers[0]), type(self.answer_candidates[0]))
            print(self.correct_answers[0] == self.answer_candidates[4])
            raise Exception()

    def __repr__(self):
        r = "Correct: " + "\n".join([a.raw_text for a in self.correct_answers])
        r += ("Candidates: " + "\n".join([c.raw_text for c in self.answer_candidates]))\
            if self.answer_candidates is not None else ""
        return r
