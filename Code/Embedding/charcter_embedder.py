import torch
from torch import nn

from Code.Training import dev
from Config.config import conf


class CharacterEmbedder(nn.Module):

    def __init__(self, dims, layers=1):
        super().__init__()
        self.out_dim = dims
        self.hidden_dim = dims//4
        if dims % 4 != 0:
            raise Exception("dims must be divisible by 4")

        self.n_layers = layers

        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, layers, batch_first=False, dropout=conf.dropout, bidirectional=True)
        self.relu = nn.ReLU()

        characters = "abcdefghijklmnopqrstuvwxyz0123456789,.&"
        self.map = {c: i for i, c in enumerate(characters)}
        self.embs = nn.Embedding(len(characters), self.hidden_dim)

    def get_character_embeddings(self, word):
        ids = [self.map[c] for c in word]
        ids = torch.tensor(ids).to(dev()).long()
        return self.embs(ids).view(len(ids), 1, -1)

    def forward(self, word):
        embs = self.get_character_embeddings(word)
        out, h = self.gru(embs, self.init_hidden(1))
        out = out.view(out.size(0), -1)
        out = torch.cat([out[0,:], out[-1,:]], dim=-1)  # head and tail concat
        return self.relu(out)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(2 * self.n_layers, batch_size, self.hidden_dim).zero_().to(dev())
        return hidden