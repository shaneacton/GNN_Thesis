import textwrap
from math import ceil

import torch

from Code.Data.Tokenisation.token_sequence import TokenSequence
from Code.Models import tokeniser, embedder


class Text:

    WRAP_TEXT_LENGTH = 150
    MAX_TOKENS = 512 # todo get max num tokens dynamically
    WINDOWED_EMBEDDINGS_OVERLAP = 50  # the amount a window looks in each direction

    def __init__(self, text):
        self.text = text
        self._token_sequence: TokenSequence=None

    @property
    def token_sequence(self) -> TokenSequence:
        if not self._token_sequence:
            self._token_sequence = TokenSequence(self.text)
        return self._token_sequence

    @property
    def clean(self):
        return " ".join(self.token_sequence.tokens)

    def get_embedding(self, sequence_reduction=None):
        """
            A special case where a given passage has more than the maximum number of tokens
            should be created. In this case the passage should be split into overlapping windows
            of maximum size. In the overlap there will be words with 2 differing contextual embeddings.
            The embedding created by the nearest window should be chosen
        """

        tokens = self.token_sequence.sub_tokens
        if len(tokens) > Text.MAX_TOKENS:
            embeddings = self.get_windowed_embeddings(tokens)
        else:
            embeddings = embedder(tokens)


        if sequence_reduction:
            return sequence_reduction(embeddings)
        return embeddings

    def get_tokens(self):
        return tokeniser(self.text)

    def __repr__(self):
        return "\n".join(textwrap.wrap(self.text, Text.WRAP_TEXT_LENGTH))

    def __eq__(self, other):
        return self.text == other.text

    def __hash__(self):
        return self.text.__hash__()

    @staticmethod
    def sum_over_sequence(seq_embedding):
        batch_size = seq_embedding.size(0)
        return torch.sum(seq_embedding, dim=1).view(batch_size,1,-1)

    @staticmethod
    def tail_concatinator(seq_embedding, cat_dim=2):
        seq_head = seq_embedding[:,0:1,:]
        num_el = seq_embedding.size(1)
        if num_el == 1:
            return torch.cat([seq_head,seq_head], dim=cat_dim)

        return torch.cat([seq_head,seq_embedding[:,num_el-1:num_el,:]], dim=cat_dim)

    @staticmethod
    def get_windowed_embeddings(tokens):
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
