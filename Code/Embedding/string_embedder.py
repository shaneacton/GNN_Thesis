import time
from abc import abstractmethod
from torch import nn

from Code.Training.timer import log_time


class StringEmbedder(nn.Module):

    @abstractmethod
    def embed(self, string, **kwargs):
        pass

    def forward(self, string, **kwargs):
        t = time.time()
        emb = self.embed(string, **kwargs)
        log_time("Token embedding", time.time() - t, increment_counter=False)
        return emb