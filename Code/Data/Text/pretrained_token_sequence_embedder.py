import threading
import time

import torch
from transformers import BertTokenizer

from Code.Training import device


class PretrainedTokenSequenceEmbedder:

    def __init__(self, embedder_type):
        start = time.time()

        self.embedder_type = embedder_type
        if embedder_type == "bert":
            from transformers import BertModel, BasicTokenizer

            print("initialising bert on thread", threading.current_thread().__class__.__name__,threading.current_thread().ident)
            self.basic_bert_tokeniser = BasicTokenizer()
            self.bert_tokeniser = BertTokenizer.from_pretrained("bert-base-uncased")

            self.bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
            print("bert initialised in", (time.time() - start), "secs")

    @property
    def embedded_size(self):
        return list(self.embed("test").size())[2]

    def basic_tokeniser(self, string):
        # without splitting into subtokens
        if self.embedder_type == "bert":
            return self.basic_bert_tokeniser.tokenize(string)

    def tokenise(self, string):
        # with subtoken splitting
        if self.embedder_type == "bert":
            return self.bert_tokeniser.tokenize(string)

    def index(self, tokens):
        # returns the emb ids
        if self.embedder_type == "bert":
            return torch.Tensor(self.bert_tokeniser.convert_tokens_to_ids(tokens)).reshape(1, -1) \
            .type(torch.LongTensor).to(device)

    def embed(self, tokens):
        if self.embedder_type == "bert":
            return self.bert_model(self.index(tokens))[0].detach()

    def __call__(self, *args, **kwargs):
        return self.embed(*args)


_tokseq_embedder = None

def tokseq_embedder():
    global _tokseq_embedder
    if _tokseq_embedder is None:
        _tokseq_embedder = PretrainedTokenSequenceEmbedder("bert")
    return _tokseq_embedder