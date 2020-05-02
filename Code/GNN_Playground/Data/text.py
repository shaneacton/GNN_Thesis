from Code.GNN_Playground.Models import tokeniser, embedder


class Text:

    def __init__(self, text):
        self.text = text

    def get_embedding(self):
        """
            A special case where a given passage has more than the maximum number of tokens
            should be created. In this case the passage should be split into overlapping windows
            of maximum size. In the overlap there will be words with 2 differing contextual embeddings.
            The embedding created by the nearest window should be chosen
        """
        max_tokens = 512  # todo get max num tokens dynamically

        tokens = tokeniser(self.text)
        if len(tokens) > max_tokens:
            # todo use window to embed full text
            print("warning, cannot embed text  with", len(tokens), "tokens")
            tokens = tokens[:max_tokens]

        return embedder(tokens)

    def get_tokens(self):
        return tokeniser(self.text)

    def __repr__(self):
        return self.text

    def __eq__(self, other):
        return self.text == other.text