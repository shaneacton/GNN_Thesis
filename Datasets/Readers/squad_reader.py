
import json
import os
from typing import Iterable

from allennlp.data import DatasetReader, Tokenizer

from Code.GNN_Playground.Data.Answers.answers import Answers
from Code.GNN_Playground.Data.Answers.extracted_answer import ExtractedAnswer
from Code.GNN_Playground.Data.context import Context
from Code.GNN_Playground.Data.passage import Passage
from Code.GNN_Playground.Data.question import Question
from Code.GNN_Playground.Data.training_example import TrainingExample
from Datasets.Readers.data_reader import DataReader


@DatasetReader.register('SQuAD')
class SQuADDatasetReader(DataReader):

    """
        SQuAD is a standard QA system
        each passage has multiple associated questions
        answers are found directly in the passages
    """

    def __init__(self, tokenizer: Tokenizer = None, token_indexers=None):
        super().__init__(tokenizer,token_indexers)

    @staticmethod
    def get_training_examples(file_path: str) -> Iterable[TrainingExample]:
        with open(file_path) as json_file:
            data = json.load(json_file)["data"]
            for example in data:
                title = example["title"]
                paragraphs = example["paragraphs"]
                # print("title:",title)
                for paragraph in paragraphs:
                    passage = Passage(paragraph["context"])
                    training_example = TrainingExample(Context(passage), title=title)
                    # print("context:",context)
                    qas = paragraph["qas"]
                    answerable_qs = [q for q in qas if not q["is_impossible"]]
                    unanswerable_qs = [q for q in qas if q["is_impossible"]]

                    context_tokens = training_example.context.get_context_tokens()

                    for qa in answerable_qs:
                        question = Question(qa["question"])
                        id = qa["id"]
                        answers_json = qa["answers"]
                        # print("q:",question)
                        # print("a's:", answers_json)
                        answer_objects = [ExtractedAnswer(a["text"], int(a["answer_start"]), context_tokens)
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

