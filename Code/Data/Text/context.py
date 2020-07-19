from typing import List, Union

import torch

from Code.Data.Text.Tokenisation.token_sequence import TokenSequence
from Code.Data.Text.passage import Passage
from Code.Data.Text.text import Text
from Code.Training import device


class Context:
    """
        a collection of passages with a natural grouping
        ie a document
    """

    PASSAGE_BREAK_STRING = "\n.\n"

    def __init__(self,passages : Union[List[Passage], Passage, None] = None):
        if isinstance(passages, List):
            self.passages : List[Passage] = passages
        if isinstance(passages, Passage):
            self.passages : List[Passage] = [passages]
        if passages is None:
            self.passages : List[Passage] = []

        self._token_sequence = None

    @property
    def token_sequence(self):
        if not self._token_sequence:
            self._token_sequence = TokenSequence(Text(self.get_full_context()))
        return self._token_sequence

    def add_passage(self, passage: Passage):
        self.passages.append(passage)

    def add_text_as_passage(self, text):
        self.passages.append(Passage(text))

    def get_all_text_pieces(self):
        return [passage.raw_text for passage in self.passages]

    def get_full_context(self):
        return Context.PASSAGE_BREAK_STRING.join(self.get_all_text_pieces())

    def get_context_embedding(self):
        """
            since the embedding used may be contextual, and thus be constrained in length
            each passage is embedded separately.
        """
        # todo possibly add in newlines between passages
        embeddings = []
        for passage in self.passages:
            embeddings.append(passage.get_embedding())
        full_embeddings = torch.cat(embeddings,dim=1)
        return full_embeddings

    def get_answer_span_vec(self, start_char_id, end_char_id):
        sub_token_span = self.token_sequence.get_word_token_span_from_chars(start_char_id, end_char_id, subtokens=True)
        sub_token_span = list(sub_token_span)
        return torch.tensor(sub_token_span).view(1, 2).to(device)




