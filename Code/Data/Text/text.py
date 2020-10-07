import textwrap
from math import ceil

import torch

from torch.multiprocessing import Pool, set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

from Code.Data.Text.Tokenisation.token_sequence import TokenSequence

_pool = None


def pool():
    global _pool
    if _pool is None:
        _pool = Pool()
    return _pool


class Text:

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
    def clean(self):
        return " ".join(self.token_sequence.raw_word_tokens)

    def __repr__(self):
        return "\n".join(textwrap.wrap(self.raw_text, Text.WRAP_TEXT_LENGTH))

    def __eq__(self, other):
        return self.raw_text == other.raw_text

    def __hash__(self):
        return self.raw_text.__hash__()

    @staticmethod
    def get_windowed_embeddings(tokens, embedder, max_tokens, overlap):
        """
        breaks the text up into appropriately sized and even chunks
        looks ahead and behind these chunks when contextually embedding
        cuts off the look-ahead/behind and stitches embeddings together
        """
        max_window_length = max_tokens - 2*overlap
        num_windows = ceil(len(tokens) / max_window_length)
        even_chunk_size = len(tokens)//num_windows  # length of windows with no overlaps

        args = ((tokens, embedder, w, num_windows, even_chunk_size, overlap) for w in range(num_windows))
        # calculates each windows embedding in parallel
        # effective_embeddings = pool().starmap(Text.get_window_embeddings, args)
        #
        # pool().close()
        # pool().join()

        effective_embeddings = []
        for arg in args:
            effective_embeddings.append(Text.get_window_embeddings(*arg))

        effective_embeddings = torch.cat(effective_embeddings, dim=1)
        if effective_embeddings.size(1) != len(tokens):
            raise Exception("Mismatch in getting windowed embeddings. given: " + str(len(tokens))
                            + " resulting: " + str(effective_embeddings.size()))
        return effective_embeddings

    @staticmethod
    def get_window_embeddings(tokens, embedder, w, num_windows, even_chunk_size, overlap):
        even_start, even_end = w * even_chunk_size, (w + 1) * even_chunk_size

        overlap_start, overlap_end = even_start - overlap, even_end + overlap
        overlap_start, overlap_end = max(overlap_start, 0), min(overlap_end, len(tokens))

        full_embeddings = embedder(tokens[overlap_start:overlap_end])

        used_start = overlap if w > 0 else 0
        used_end = -overlap if w < num_windows - 1 else (overlap_end - overlap_start)
        used_embeddings = full_embeddings[:, used_start:used_end, :]
        return used_embeddings
