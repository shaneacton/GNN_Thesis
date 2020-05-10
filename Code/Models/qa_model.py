from torch import nn

from Code.Data.context import Context
from Code.Data.question import Question
from Datasets.Batching.batch import Batch


class QAModel(nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_context_query_candidates_vecs(*args):
        """

        :param args: may be a batch, or a (context, question) tuple
        :return: (context, query, Optional[candidates]) vecs
        """
        if len(args) == 1:
            if type(args[0]) == Batch:
                batch: Batch = args[0]
                return batch.get_cqc_vecs()

        if len(args) >= 2 and type(args[0]) == Context and type(args[1]) == Question:
            context: Context = args[0]
            question: Question = args[1]
            return context.get_context_embedding(), question.get_embedding(), question.get_candidates_embedding()

        raise Exception("unrecognised args: " + repr(args))

    @staticmethod
    def get_answer_type(*args):
        if len(args) == 1 and type(args[0]) == Batch:
            batch: Batch = args[0]
            return batch.get_answer_type()

        if len(args) >= 2 and type(args[1]) == Question:
            question: Question = args[1]
            return question.get_answer_type()

        raise Exception("unrecognised args: " + repr(args) + " type:" +repr(type(args[0])))