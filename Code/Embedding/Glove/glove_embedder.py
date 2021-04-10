from os.path import join, exists

import numpy as np
import torch
from torch import Tensor
import re
import pathlib

from torch.nn import LayerNorm

from Code.Embedding.charcter_embedder import CharacterEmbedder
from Code.Embedding.positional_embedder import PositionalEmbedder
from Code.Embedding.string_embedder import StringEmbedder
from Code.Training import dev
from Config.config import conf


class GloveEmbedder(StringEmbedder):

    def __init__(self, use_positional_embeddings=True):
        super().__init__()
        self.use_positional_embeddings = use_positional_embeddings
        file_path = pathlib.Path(__file__).parent.absolute()
        print("glove path:", file_path)
        embeddings_dict = {}
        self.dims = glove_dims = conf.embedded_dims
        self.use_character_embeddings = conf.character_embedded_dims > 0
        if self.use_character_embeddings:
            """
                must use a character level word embedding. 
                the final embedding will be the feature-wise concat of the glove and character embeddings
            """
            glove_dims -= conf.character_embedded_dims
            self.full_character_embedder = CharacterEmbedder(conf.character_embedded_dims)

        path = join(file_path, "glove.6B", "glove.6B." + repr(glove_dims) + "d.txt")
        if not exists(path):
            raise Exception("no glove embeddings of dimension " + repr(glove_dims) + " emb dim: " + repr(self.dims) +
                            " character dim: " + repr(conf.character_embedded_dims) + ". glove dim = emb-character")
        with open(path, 'r', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector

        self.regex = re.compile('[^a-zA-Z 0123456789,.&]')

        self.embs = embeddings_dict

        if use_positional_embeddings:
            self.positional_embedder = PositionalEmbedder()

        if conf.use_layer_norms_b:
            self.norm = LayerNorm(self.dims)

        if conf.use_character_embs_for_unknown_words:
            self.character_embedder = CharacterEmbedder(glove_dims)
        else:
            self.unknown_token_emb = np.asarray([0] * glove_dims, "float32")

    def get_emb(self, word, allow_unknowns=True):
        if word in self.embs.keys():
            emb = self.embs[word]
        else:
            if conf.use_character_embs_for_unknown_words:
                emb = self.character_embedder(word)
            else:
                emb = self.unknown_token_emb

        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb).to(dev())

        try:  # todo remove legacy
            if self.use_character_embeddings:
                c_emb = self.full_character_embedder(word)
                emb = torch.cat([emb, c_emb], dim=-1)
        except:
            pass
        return emb

    def get_words(self, string):
        string = string.replace(",", " , ")
        string = string.replace("&", " & ")
        string = string.replace(".", "  ")
        string = string.replace("'s", "")
        string = string.replace("'", "")

        string = self.regex.sub(' ', string)  # remove all non alpha numeric characters
        words = string.lower().split()
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
            if conf.use_layer_norms_b:
                embs = self.norm(embs)

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