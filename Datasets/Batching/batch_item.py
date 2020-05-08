from Code.GNN_Playground.Data.data_sample import DataSample
from Code.GNN_Playground.Data.question import Question


class BatchItem:

    """
        a context/question pair
    """

    def __init__(self, data_example: DataSample, question: Question):
        self.data_example: DataSample = data_example
        self.question: Question = question

