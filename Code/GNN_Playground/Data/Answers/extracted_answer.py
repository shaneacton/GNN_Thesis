from Code.GNN_Playground.Data.Answers.answers import Answer


class ExtractedAnswer(Answer):
    """
        an extracted answer is a direct rip from the context
        it can be expressed as a start and end char index pair
    """

    def __init__(self, text, start_id):
        super().__init__(text)
        self.start_id = start_id
        self.end_id = start_id + len(text)