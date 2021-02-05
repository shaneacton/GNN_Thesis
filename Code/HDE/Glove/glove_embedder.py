from os.path import join

import numpy as np
import torch
from torch import Tensor, nn
import re
import pathlib

from Code.Training import device


class GloveEmbedder(nn.Module):

    def __init__(self, dims=50, max_positions=4050, use_positional_embeddings=True):
        super().__init__()
        self.use_positional_embeddings = use_positional_embeddings
        file_path = pathlib.Path(__file__).parent.absolute()
        print("file path:", file_path)
        embeddings_dict = {}
        self.dims = dims
        path = join(file_path, "glove.6B", "glove.6B." + repr(self.dims) + "d.txt")
        with open(path, 'r', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector

        self.unknown_token_emb = np.asarray([0] * self.dims, "float32")
        self.regex = re.compile('[^a-zA-Z 0123456789,.&]')

        self.embs = embeddings_dict

        self.positional_embs = nn.Embedding(max_positions, dims)

    def get_emb(self, word):
        if word in self.embs.keys():
            emb = self.embs[word]
        else:
            emb = self.unknown_token_emb
        return torch.tensor(emb)

    def get_words(self, string):
        string = string.replace(",", " , ")
        string = string.replace("&", " & ")
        string = string.replace(".", "  ")
        string = string.replace("'s", "")
        string = string.replace("'", "")

        string = self.regex.sub(' ', string)  # remove all non alpha numeric characters
        words = string.lower().split()
        return words

    def embed(self, string: str) -> Tensor:
        words = self.get_words(string)
        if len(words) == 0:
            print("num words:", len(words), "string:", string, "words:", words)
            raise NoWordsException("no words from string " + string)
        embs = []
        for w in words:
            tens = self.get_emb(w)
            if tens.size(0) != self.dims:
                out = repr(self.embs[w]) if w in self.embs else "uknown"
                raise Exception("word: " + w + " emb: " + repr(tens.size()) + " map: " + out)
            embs.append(tens.view(1, self.dims))

        seq_len = len(embs)
        embs = torch.cat(embs, dim=0).view(1, seq_len, -1)
        embs = embs.to(device)
        if self.use_positional_embeddings:
            pos_ids = torch.tensor([i for i in range(seq_len)]).long().to(device)
            pos_embs = self.positional_embs(pos_ids).view(1, seq_len, -1)
            embs += pos_embs

        # print("emb:", embs.size())
        # print(embs)
        return embs

    def forward(self, string):
        return self.embed(string)


class NoWordsException(Exception):
    pass


if __name__ == "__main__":
    embedder = GloveEmbedder()
    embs = embedder.embed("hello world. shmeg 7")
    print("words:", embedder.get_words("hello world. shmeg 7"))
    print("embs:", embs)

    print("&" in embedder.embs.keys())

    text = 'Its official motto is "LÉtoile du Nord" (French: "Star of the North").Minnesota'
    print(embedder.get_words(text))

    print(embedder(text))