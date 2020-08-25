from Code.Data.Text.text import Text


class Answer(Text):

    def __init__(self, text: str):
        super().__init__(text)

    def get_output_model(self):
        raise NotImplementedError()