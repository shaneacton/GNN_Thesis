import textwrap
from math import ceil
from typing import List

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
    def get_windowed_embeddings(all_tokens, embedder, max_tokens, overlap):
        """
        breaks the text up into appropriately sized and even chunks
        looks ahead and behind these chunks when contextually embedding
        cuts off the look-ahead/behind and stitches embeddings together
        """
        num_tokens = len(all_tokens)

        max_window_length = max_tokens - 2*overlap
        num_windows = ceil(num_tokens / max_window_length)
        even_chunk_size = num_tokens//num_windows  # length of windows with no overlaps

        batch_size = 5
        num_batches = ceil(num_windows/batch_size)
        batches = [[] for _ in range(num_batches)]
        for w in range(num_windows):  # group the windows into batches
            window, overlap_start, overlap_end = Text.get_window_tokens(all_tokens, w, even_chunk_size, overlap)
            batch_num = w//batch_size
            batches[batch_num].append(window)

        w=0
        effective_embeddings = []
        for b in range(num_batches):  # do batchwise encoding, then chop off overlaps
            batch: List[List[str]] = batches[b]
            batch_embeddings = embedder(batch)  # is of shape (batch, padded_len, features)
            # need of shape (num_tokens, features)
            for i in range(len(batch)):
                embs = batch_embeddings[i,:len(batch[i]),:]  # cut off padding
                # cut off overlaps
                embs = Text.get_used_embeddings(embs, num_tokens, w, num_windows, even_chunk_size, overlap)
                effective_embeddings.append(embs)
                w += 1

        effective_embeddings = torch.cat(effective_embeddings)  # stitch embeddings back together
        if effective_embeddings.size(0) != len(all_tokens):
            raise Exception("Mismatch in getting windowed embeddings. given: " + str(len(all_tokens))
                            + " resulting: " + str(effective_embeddings.size()))
        effective_embeddings = effective_embeddings.view(1, num_tokens, -1)
        return effective_embeddings

    @staticmethod
    def get_window_tokens(all_tokens, w, even_chunk_size, overlap):
        """returns the tokens in the overlapping window"""
        overlap_start, overlap_end = Text.get_overlap_range(len(all_tokens), w, even_chunk_size, overlap)
        return all_tokens[overlap_start:overlap_end], overlap_start, overlap_end

    @staticmethod
    def get_overlap_range(num_tokens,  w, even_chunk_size, overlap):
        even_start, even_end = w * even_chunk_size, (w + 1) * even_chunk_size

        overlap_start, overlap_end = even_start - overlap, even_end + overlap
        overlap_start, overlap_end = max(overlap_start, 0), min(overlap_end, num_tokens)
        return overlap_start, overlap_end

    @staticmethod
    def get_used_embeddings(windowed_embeddings, num_tokens, w, num_windows, even_chunk_size, overlap):
        """chops off the unused embeddings from the overlaps"""
        overlap_start, overlap_end = Text.get_overlap_range(num_tokens, w, even_chunk_size, overlap)

        used_start = overlap if w > 0 else 0
        used_end = -overlap if w < num_windows - 1 else (overlap_end - overlap_start)
        used_embeddings = windowed_embeddings[used_start:used_end, :]
        return used_embeddings


