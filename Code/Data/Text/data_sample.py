import random
from typing import List

from Code.Data.Text.Answers.answers import Answers
from Code.Data.Text.Answers.extracted_answer import ExtractedAnswer
from Code.Data.Text.context import Context
from Code.Data.Text.question import Question


class DataSample:

    """
        one context or passage may contain multiple questions
        each question may contain multiple correct answers
        as well as multiple answer candidates
    """

    def __init__(self, context : Context, questions=None, title=""):
        self.context = context
        self.title = title

        if isinstance(questions, List):
            self.questions: List[Question] = questions
        if isinstance(questions, Question):
            self.questions: List[Question] = [questions]
        if questions is None:
            self.questions: List[Question] = []

    @property
    def title_and_peek(self):
        peek = "_".join(self.context.passages[0].token_sequence.raw_word_tokens[:4])
        return self.title + ("_" if self.title else "") + peek

    def add_question(self, question : Question):
        self.questions.append(question)

    def get_all_text_pieces(self):
        return self.context.get_all_text_pieces() + [text for q in self.questions for text in q.get_all_text_pieces()] \
               + ([self.title] if self.title else [])

    def get_answer_span_vec(self, answers: Answers):
        """
        uses random samping of the multiple correct span answers
        :return:
        """
        if self.get_answer_type() != ExtractedAnswer:
            raise Exception()
        # todo closest match instead of random
        remaining_answers = set(answers.correct_answers)
        while remaining_answers:
            answer: ExtractedAnswer = random.choice(list(remaining_answers))
            try:
                return self.context.get_answer_span_vec(answer.start_char_id, answer.end_char_id)
            except:
                remaining_answers-= {answer}  # answer may be broken
        raise Exception("failed to get subtokens for all answers")

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

    def get_answer_type(self):
        return self.questions[0].get_answer_type()

    def get_output_model(self):
        return self.questions[0].answers.get_output_model()