
import json
import os
from typing import Iterable

from Code.Data.Text.Answers.answers import Answers
from Code.Data.Text.Answers.extracted_answer import ExtractedAnswer
from Code.Data.Text.context import Context
from Code.Data.Text.passage import Passage
from Code.Data.Text.question import Question
from Code.Data.Text.data_sample import DataSample
from Datasets.Readers.data_reader import DataReader


class SQuADDatasetReader(DataReader):

    """
        SQuAD is a standard QA system
        each passage has multiple associated questions
        answers are found directly in the passages
    """

    def get_dev_set(self):
        return self.get_data_samples(self.dev_set_location())

    def get_data_samples(self, file_path: str) -> Iterable[DataSample]:
        with open(file_path) as json_file:
            data = json.load(json_file)["data"]
            for example in data:
                title = example["title"]
                paragraphs = example["paragraphs"]
                for paragraph in paragraphs:
                    passage = Passage(paragraph["context"])
                    training_example = DataSample(Context(passage), title=title)
                    qas = paragraph["qas"]
                    answerable_qs = [q for q in qas if not q["is_impossible"]]
                    unanswerable_qs = [q for q in qas if q["is_impossible"]]

                    for qa in answerable_qs:
                        question = Question(qa["question"])
                        id = qa["id"]
                        answers_json = qa["answers"]
                        answer_objects = [ExtractedAnswer(a["text"], int(a["answer_start"]))
                                          for a in answers_json]
                        answers = Answers(answer_objects)
                        question.answers = answers
                        training_example.add_question(question)

                    yield training_example

    @staticmethod
    def dev_set_location():
        return os.path.join(DataReader.raw_data_location(), "SQuAD", "dev-v2.0.json")

    @staticmethod
    def train_set_location():
        return os.path.join(DataReader.raw_data_location(), "SQuAD", "train-v2.0.json")

