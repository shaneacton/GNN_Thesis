import torch

from Code.GNN_Playground.Data.Tokenisation.token_sequence import TokenSequence
from Code.GNN_Playground.Models import tokeniser, embedder


class Text:

    def __init__(self, text):
        self.text = text
        self._token_sequence: TokenSequence=None

    @property
    def token_sequence(self):
        if not self._token_sequence:
            self._token_sequence = TokenSequence(self.text)
        return self._token_sequence

    def get_embedding(self, sequence_reduction=None):
        """
            A special case where a given passage has more than the maximum number of tokens
            should be created. In this case the passage should be split into overlapping windows
            of maximum size. In the overlap there will be words with 2 differing contextual embeddings.
            The embedding created by the nearest window should be chosen
        """
        max_tokens = 512  # todo get max num tokens dynamically

        tokens = self.token_sequence.flat_sub_tokens
        if len(tokens) > max_tokens:
            # todo use window to embed full text
            print("warning, cannot embed text  with", len(tokens), "tokens")
            tokens = tokens[:max_tokens]

        # subtoken_map = subtoken_mapper(self.text)
        # print(subtoken_map)
        if sequence_reduction:
            return sequence_reduction(embedder(tokens))
        return embedder(tokens)

    def get_tokens(self):
        return tokeniser(self.text)

    def __repr__(self):
        return self.text

    def __eq__(self, other):
        return self.text == other.text

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