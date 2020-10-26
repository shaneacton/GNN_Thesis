from typing import Callable, List

import torch
from torch import nn

from Code.Config import GraphEmbeddingConfig
from Code.Data.Text.Tokenisation.token_sequence import TokenSequence
from Code.Data.Text.pretrained_token_sequence_embedder import PretrainedTokenSequenceEmbedder
from Code.Data.Text.text import Text


class TokenSequenceEmbedder(nn.Module):
    """
    wrapper around any pretrained contextual embedder such as bert
    """

    def __init__(self, gec: GraphEmbeddingConfig, token_indexer=None, index_embedder=None, token_embedder=None):
        super().__init__()
        self.gec = gec
        self.token_embedder : PretrainedTokenSequenceEmbedder = token_embedder # raw tokens into embeddings
        self.token_indexer: Callable[[List[str]], List[torch.Tensor]] = token_indexer  # raw tokens into ids
        # converts token  ids to embedded vectors
        self.index_embedder: Callable[[List[torch.Tensor]], List[torch.Tensor]] = index_embedder

    def get_embedded_sequence(self, seq: TokenSequence):
        tokens = seq.raw_subtokens
        if self.token_embedder:
            max_seq = self.gec.max_bert_token_sequence
            overlap = self.gec.bert_window_overlap_tokens
            return Text.get_windowed_embeddings(tokens, self.token_embedder, max_seq, overlap)

        indexes = self.token_indexer(seq.raw_subtokens)
        return self.index_embedder(indexes)

    def forward(self, seq: TokenSequence):
        if self.gec.fine_tune_token_embedder:
            return self.get_embedded_sequence(seq)

        with torch.no_grad():
            return self.get_embedded_sequence(seq)