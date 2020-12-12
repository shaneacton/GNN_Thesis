from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor
from transformers import BatchEncoding

from Code.Training import device


class Embedder(nn.Module, ABC):

    @abstractmethod
    def embed(self, input_ids, attention_mask):
        raise NotImplementedError()

    def forward(self, encoding: BatchEncoding):
        input_ids = Tensor(encoding["input_ids"]).type(torch.LongTensor).to(device)
        attention_mask = Tensor(encoding["attention_mask"]).type(torch.LongTensor).to(device)
        return self.embed(input_ids, attention_mask)
