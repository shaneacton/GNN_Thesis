from typing import List

from Code.Data.context import Context
from Code.Data.question import Question


class DataSample:

    """
        one context or passage may contain multiple questions
        each question may contain multiple correct answers
        as well as multiple answer candidates
    """

    def __init__(self, context : Context, questions=None, title=None):
        self.context = context
        self.title = title

        if isinstance(questions, List):
            self.questions = questions
        if isinstance(questions, Question):
            self.questions = [questions]
        if questions is None:
            self.questions = []

    def add_question(self, question : Question):
        self.questions.append(question)

    def get_all_text_pieces(self):
        return self.context.get_all_text_pieces() + [text for q in self.questions for text in q.get_all_text_pieces()] \
               + ([self.title] if self.title else [])

    def __repr__(self):
        rep = ""
        if self.title:
            rep += "title: " + self.title + "\n\n"

        rep += "context:\n"

        passages = [repr(passage) for passage in self.context.passages]
        rep += "\n\n".join(passages) + "\n\n"

        rep += "question" + (":" if len(self.questions) == 1 else "s:")

        for question in self.questions:
            rep += "\n\nQ: " + repr(question) + "\n"

            if question.answers.answer_candidates:
                rep += "candidates:\n"
                candidates = set([ " * " + repr(can) for can in question.answers.answer_candidates])
                rep += "\n".join(candidates) + "\n"

            answers = set([repr(ans) for ans in question.answers.correct_answers])
            answers = [(" * " if len(answers) > 1 else "") + ans for ans in answers  ]

            rep += "A: " if len(answers) == 1 else "Valid answers:\n"
            rep += "\n".join(answers) + "\n"

        return rep