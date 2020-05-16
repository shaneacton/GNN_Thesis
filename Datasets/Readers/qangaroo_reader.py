import json
import os
from typing import Iterable


from Code.Data.Text.Answers.answers import Answers
from Code.Data.Text.Answers.candidate_answer import CandidateAnswer
from Code.Data.Text.context import Context
from Code.Data.Text.question import Question
from Code.Data.Text.data_sample import DataSample
from Datasets.Readers.data_reader import DataReader


class QUangarooDatasetReader(DataReader):

    """
        QUangaroo is a combination of wikihop and medhop
        each is a multihop, multiple choice answering set
        QUangaroo pairs 1 question with multiple passages
    """

    def get_dev_set(self, set_name="wikihop"):
        return self.get_data_samples(self.dev_set_location(set_name))

    def get_data_samples(self, file_path: str) -> Iterable[DataSample]:
        with open(file_path) as json_file:
            data = json.load(json_file)
            for question_data in data:
                candidates = question_data["candidates"]
                answer = question_data["answer"]
                answer_object = Answers(CandidateAnswer(answer), [CandidateAnswer(candidate) for candidate in candidates])

                query = question_data["query"]
                question = Question(query, answer_object)

                supports = question_data["supports"]  # here supports is an array of passages

                context = Context()
                for support in supports:
                    context.add_text_as_passage(support)

                training_example = DataSample(context, question)
                yield training_example


    @staticmethod
    def qangaroo_set_location():
        return os.path.join(DataReader.raw_data_location(), "qangaroo_v1.1")

    @staticmethod
    def dev_set_location(set_name="wikihop"):
        return os.path.join(QUangarooDatasetReader.qangaroo_set_location(), set_name, "dev.json")

    @staticmethod
    def train_set_location(set_name):
        return os.path.join(QUangarooDatasetReader.qangaroo_set_location(), set_name, "train.json")