from Code.Data.Text.data_sample import DataSample
from Code.Data.Text.question import Question


class BatchItem:

    """
        a context/question pair
    """

    def __init__(self, data_example: DataSample, question: Question):
        self.data_sample: DataSample = data_example
        self.question: Question = question

    def __repr__(self):
        # todo exlude other questions
        return repr(self.data_sample)


