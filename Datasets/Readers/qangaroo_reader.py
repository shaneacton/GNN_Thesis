import json
import os
from typing import Iterable

from allennlp.data import DatasetReader, Tokenizer

from Code.GNN_Playground.Data.Answers.answers import Answers
from Code.GNN_Playground.Data.Answers.one_word_answer import OneWordAnswer
from Code.GNN_Playground.Data.context import Context
from Code.GNN_Playground.Data.question import Question
from Code.GNN_Playground.Data.training_example import TrainingExample
from Datasets.Readers.data_reader import DataReader


@DatasetReader.register('QAngaroo')
class QUangarooDatasetReader(DataReader):

    """
        QUangaroo is a combination of wikihop and medhop
        each is a multihop, multiple choice answering set
        QUangaroo pairs 1 question with multiple passages
    """

    def __init__(self, tokenizer: Tokenizer=None, token_indexers=None):
        super().__init__(tokenizer,token_indexers)

    @staticmethod
    def get_training_examples(file_path: str) -> Iterable[TrainingExample]:
        with open(file_path) as json_file:
            data = json.load(json_file)
            for question_data in data:
                candidates = question_data["candidates"]
                answer = question_data["answer"]
                answer_object = Answers(OneWordAnswer(answer), [OneWordAnswer(candidate) for candidate in candidates])

                query = question_data["query"]
                question = Question(query, answer_object)

                supports = question_data["supports"]  # here supports is an array of passages

                context = Context()
                for support in supports:
                    context.add_text_as_passage(support)

                training_example = TrainingExample(context, question)
                yield training_example


    @staticmethod
    def qangaroo_set_location():
        return os.path.join(DataReader.raw_data_location(), "QAngaroo")

    @staticmethod
    def dev_set_location(set_name):
        return os.path.join(QUangarooDatasetReader.qangaroo_set_location(), set_name, "dev.json")

    @staticmethod
    def train_set_location(set_name):
        return os.path.join(QUangarooDatasetReader.qangaroo_set_location(), set_name, "train.json")