from Code.Embedding.bert_embedder import BertEmbedder
from Code.Utils import graph_utils
from Code.HDE.hde_model import HDEModel
from Config.config import conf


class HDEBert(HDEModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedder = BertEmbedder()
        self.embedder_name = "bert"
        from Code.Utils.model_utils import num_params
        conf.cfg["num_embedding_params"] = num_params(self.embedder)
        conf.cfg["num_total_params"] += conf.cfg["num_embedding_params"]

    def create_graph(self, example):
        return graph_utils.create_graph(example, tokeniser=self.embedder.tokenizer)