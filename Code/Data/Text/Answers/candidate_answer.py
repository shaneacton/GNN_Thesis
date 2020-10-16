from Code.Data.Text.Answers.answer import Answer


class CandidateAnswer(Answer):

    def __init__(self, text: str):
        super().__init__(text)