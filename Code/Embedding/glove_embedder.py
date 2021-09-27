from os.path import join, exists

import numpy as np
import torch
from torch import Tensor
import re
import pathlib

from torch.nn import LayerNorm

from Code.Embedding.character_embedder import CharacterEmbedder
from Code.Embedding.positional_embedder import PositionalEmbedder
from Code.Embedding.string_embedder import StringEmbedder
from Code.Training import dev
from Config.config import conf


class GloveEmbedder(StringEmbedder):

    def __init__(self, use_positional_embeddings=True):
        super().__init__()
        self.use_positional_embeddings = use_positional_embeddings
        embeddings_dict = {}
        self.dims = glove_dims = conf.embedded_dims
        glove_code = "glove." + conf.glove_tokens_code + "." + repr(glove_dims) + "d.txt"
        if conf.run_args.glove_path:
            path = join(conf.run_args.glove_path, glove_code)
        else:
            file_path = pathlib.Path(__file__).parent.absolute()
            path = join(file_path, "Glove", glove_code)
        print("loading glove. path:", path)

        if not exists(path):
            raise Exception("no glove embeddings of dimension " + repr(glove_dims) + " emb dim: " + repr(self.dims))
        with open(path, 'r', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                # print("word:", word, "vec:", values[1:])
                try:
                    vector = np.asarray(values[1:], "float32")
                except:  # some weird characters in larger glove files
                    continue
                embeddings_dict[word] = vector

        self.regex = re.compile('[^a-zA-Z 0123456789,.&]')

        self.embs = embeddings_dict

        if use_positional_embeddings:
            self.positional_embedder = PositionalEmbedder()

        if conf.use_character_embs_for_unknown_words:
            self.unknown_character_embedder = CharacterEmbedder(glove_dims)
        else:
            self.unknown_token_emb = np.asarray([0] * glove_dims, "float32")

    def get_emb(self, word):
        if word in self.embs.keys():
            emb = self.embs[word]
        else:
            if conf.use_character_embs_for_unknown_words:
                emb = self.unknown_character_embedder(word)
            else:
                emb = self.unknown_token_emb

        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb).to(dev())

        return emb

    def get_words(self, string, cased=False):
        string = string.replace(",", " , ")
        string = string.replace("&", " & ")
        string = string.replace(".", "  ")
        string = string.replace("'s", "")
        string = string.replace("'", "")

        string = self.regex.sub(' ', string)  # remove all non alpha numeric characters
        if not cased:
            string = string.lower()
        words = string.split()
        return words

    def embed(self, string: str, **kwargs) -> Tensor:
        words = self.get_words(string)
        if len(words) == 0:
            print("num words:", len(words), "string:", string, "words:", words)
            raise NoWordsException("no words from string " + string)
        embs = []
        for w in words:
            tens = self.get_emb(w, **kwargs)
            if tens.size(0) != self.dims:
                out = repr(self.embs[w]) if w in self.embs else "unknown"
                raise Exception("word: " + w + " emb: " + repr(tens.size()) + " map: " + out)
            embs.append(tens.view(1, self.dims))

        seq_len = len(embs)
        embs = torch.cat(embs, dim=0).view(1, seq_len, -1)
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