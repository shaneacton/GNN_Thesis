from typing import Iterable

from Code.GNN_Playground.Data.data_sample import DataSample
from Datasets.Readers.data_reader import DataReader


class Dataset:

    def __init__(self, data_reader: DataReader):
        self.data_reader=data_reader
        self.training_examples = []

    def add_example(self, example):
        self.training_examples.append(example)

    def get_training_examples(self, file_path: str, batch_size) -> Iterable[DataSample]:
