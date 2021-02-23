from os.path import join

import numpy as np
import torch
from torch import Tensor, nn
import re
import pathlib

from Code.Config.config import config
from Code.Embedding.positional_embedder import PositionalEmbedder
from Code.Embedding.string_embedder import StringEmbedder
from Code.Training import device


class GloveEmbedder(StringEmbedder):

    def __init__(self, use_positional_embeddings=True):
        super().__init__()
        self.use_positional_embeddings = use_positional_embeddings
        file_path = pathlib.Path(__file__).parent.absolute()
        print("glove path:", file_path)
        embeddings_dict = {}
        self.dims = config.embedded_dims
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

        if use_positional_embeddings:
            self.positional_embedder = PositionalEmbedder()

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
            pos_embs = self.positional_embedder.get_pos_embs(seq_len)
            embs += pos_embs

        return embs


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