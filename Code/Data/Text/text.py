import textwrap
from math import ceil

import torch

from Code.Data.Text.Tokenisation.token_sequence import TokenSequence
from Code.Data import embedder


class Text:

    WRAP_TEXT_LENGTH = 150
    MAX_TOKENS = 512 # todo get max num tokens dynamically
    WINDOWED_EMBEDDINGS_OVERLAP = 50  # the amount a window looks in each direction

    CACHE_EMBEDDING = True

    def __init__(self, text):
        self.raw_text = text
        self._token_sequence: TokenSequence = None
        self._full_embedding: torch.Tensor = None

    @property
    def token_sequence(self) -> TokenSequence:
        if self._token_sequence is None:
            self._token_sequence = TokenSequence(self)
        return self._token_sequence

    @property
    def full_embedding(self):
        if Text.CACHE_EMBEDDING:
            if self._full_embedding is None:
                self._full_embedding = self.get_embedding()
            return self._full_embedding
        else:
            return self.get_embedding()

    @property
    def clean(self):
        return " ".join(self.token_sequence.raw_word_tokens)

    def get_embedding(self, sequence_reduction=None):
        """
            In the case where the text has more than the maximum number of tokens allowed by
            the contextual embedder - the token sequence is split into overlapping windows
            each of which is embedded separately and then combined
        """

        if not self._full_embedding:
            tokens = self.token_sequence.raw_subtokens
            if len(tokens) > Text.MAX_TOKENS:
                full_embeddings = self.get_windowed_embeddings(tokens)
            else:
                full_embeddings = embedder(tokens)
        else:
            # stored in memory so it can be called multiple times with different reductions
            # without re-embedding
            full_embeddings = self._full_embedding

        if sequence_reduction:
            return sequence_reduction(full_embeddings)
        return full_embeddings

    def __repr__(self):
        return "\n".join(textwrap.wrap(self.raw_text, Text.WRAP_TEXT_LENGTH))

    def __eq__(self, other):
        return self.raw_text == other.raw_text

    def __hash__(self):
        return self.raw_text.__hash__()

    @staticmethod
    def get_windowed_embeddings(tokens, embedder=embedder):
        """
        breaks the text up into appropriately sized and even chunks
        looks ahead and behind these chunks when contextually embedding
        cuts off the look-ahead/behind and stitches embeddings together
        """
        overlap=Text.WINDOWED_EMBEDDINGS_OVERLAP
        max_window_length = Text.MAX_TOKENS - 2*overlap
        num_windows = ceil(len(tokens) / max_window_length)
        even_chunk_size = len(tokens)//num_windows  # length of windows with no overlaps

        effective_embeddings = []
        for w in range(num_windows):
            even_start, even_end = w * even_chunk_size, (w+1) * even_chunk_size

            overlap_start, overlap_end = even_start - overlap, even_end + overlap
            overlap_start, overlap_end = max(overlap_start, 0), min(overlap_end, len(tokens))

            full_embeddings = embedder(tokens[overlap_start:overlap_end])

            used_start = overlap if w>0 else 0
            used_end = -overlap if w<num_windows-1 else (overlap_end-overlap_start)
            used_embeddings = full_embeddings[:,used_start:used_end,:]
            effective_embeddings.append(used_embeddings)

        effective_embeddings = torch.cat(effective_embeddings, dim=1)
        if effective_embeddings.size(1) != len(tokens):
            raise Exception("Mismatch in getting windowed embeddings. given: " + str(len(tokens))
                            + " resulting: " + str(effective_embeddings.size()))
        return effective_embeddings
