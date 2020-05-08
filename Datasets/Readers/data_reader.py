import os
from typing import Iterable

from allennlp.data import DatasetReader, Tokenizer
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data import DatasetReader, Instance, Tokenizer

from Code.GNN_Playground.Data.data_sample import DataSample


class DataReader(DatasetReader):

    def __init__(self, tokenizer: Tokenizer, token_indexers=None):
        super().__init__()
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def _read(self, file_path: str) -> Iterable[Instance]:
        for training_example in self.get_training_examples(file_path):
            for text_piece in training_example.get_all_text_pieces():
                yield self.text_to_instance(text_piece)

    def text_to_instance(self, string: str, label=None) -> Instance:
        fields = {}
        tokens = self._tokenizer.tokenize(string)
        fields['tokens'] = TextField(tokens, self._token_indexers)
        if label:
            fields['label'] = LabelField(label, skip_indexing=True)
        return Instance(fields)

    @staticmethod
    def get_training_examples(file_path: str) -> Iterable[DataSample]:
        pass

    @staticmethod
    def raw_data_location():
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..","Raw")