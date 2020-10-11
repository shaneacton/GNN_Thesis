from Code.Data.Text.Answers.answers import Answer


class ExtractedAnswer(Answer):
    """
        an extracted answer is a direct rip from the context
        it can be expressed as a start and end char index pair
    """

    def __init__(self, text, start_char_id):
        super().__init__(text)
        self.start_char_id = start_char_id
        self.end_char_id = start_char_id + len(text)

        self.start_token_id = None
        self.end_token_id = None

    def __repr__(self):
        return self.raw_text + "\t\t- char ids : [" + repr(self.start_char_id) + ", " + repr(self.end_char_id) + "]"

    def get_output_model(self):
        raise Exception()
