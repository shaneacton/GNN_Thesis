import threading
import time
from typing import List

import torch
from transformers import BertTokenizer

from Code.Training import device


class PretrainedTokenSequenceEmbedder:

    def __init__(self, gec):
        start = time.time()

        self.gec = gec
        if gec.token_embedder_type == "bert":
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
        if self.gec.token_embedder_type == "bert":
            return self.basic_bert_tokeniser.tokenize(string)

    def tokenise(self, string):
        # with subtoken splitting
        if self.gec.token_embedder_type == "bert":
            return self.bert_tokeniser.tokenize(string)

    def index(self, tokens):
        # returns the emb ids, handles batch of token lists or single token list
        if isinstance(tokens[0], List):
            # batch mode
            batch_size = len(tokens)
        else:
            batch_size = 1
            tokens = [tokens]

        if self.gec.token_embedder_type == "bert":

            indexes = []
            max_len = max([len(toks) for toks in tokens])
            for i in range(batch_size):
                ids = torch.Tensor(self.bert_tokeniser.convert_tokens_to_ids(tokens[i]))
                ids = torch.cat([ids, torch.zeros(max_len-len(tokens[i]))])  # pad
                indexes.append(ids)
            indexes = torch.stack(indexes)

        return indexes.reshape(batch_size, -1).type(torch.LongTensor).to(device)

    def embed(self, tokens):
        ids = self.index(tokens)
        if self.gec.token_embedder_type == "bert":
            if self.gec.use_contextual_embeddings:
                embedder = lambda x: self.bert_model(x)[0]
            else:
                embedder = self.bert_model.embeddings

        if self.gec.fine_tune_token_embedder:
            embs = embedder(ids)
        else:
            with torch.no_grad():
                embs = embedder(ids)
            embs.detach()
        return embs

    def __call__(self, *args, **kwargs):
        return self.embed(*args)


_tokseq_embedder = None

def tokseq_embedder():
    global _tokseq_embedder
    if _tokseq_embedder is None:
        from Code.Config import gec
        _tokseq_embedder = PretrainedTokenSequenceEmbedder(gec)
    return _tokseq_embedder