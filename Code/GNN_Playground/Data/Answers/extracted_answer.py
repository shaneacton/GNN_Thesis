from Code.GNN_Playground.Data.Answers.answers import Answer


class ExtractedAnswer(Answer):
    """
        an extracted answer is a direct rip from the context
        it can be expressed as a start and end char index pair
    """

    def __init__(self, text, start_char_id):
        super().__init__(text)
        self.start_char_id = start_char_id
        self.end_char_id = start_char_id + len(text)