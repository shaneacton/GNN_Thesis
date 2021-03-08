from Code.Embedding.bert_embedder import BertEmbedder
from Code.HDE.Graph import graph_utils
from Code.HDE.hde_model import HDEModel


class HDEBert(HDEModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedder = BertEmbedder()
        self.embedder_name = "bert"

    def create_graph(self, example):
        return graph_utils.create_graph(example, tokeniser=self.embedder.tokenizer)