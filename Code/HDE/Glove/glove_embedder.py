from os.path import join

import numpy as np
import torch
from torch import Tensor
import re
import pathlib

from Code.Training import device


class GloveEmbedder:

    def __init__(self, dims=50):
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

    def __call__(self, string):
        return self.embed(string)

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


        embs = torch.cat(embs, dim=0).view(1, len(embs), -1)

        # print("emb:", embs.size())
        # print(embs)
        return embs.to(device)


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