from Code.Embedding.Glove.glove_embedder import GloveEmbedder
from Code.HDE.Graph import graph_utils
from Code.HDE.hde_model import HDEModel
from Config.config import conf


class HDEGlove(HDEModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedder = GloveEmbedder()
        self.embedder_name = "glove"
        from Code.Training.Utils.model_utils import num_params
        conf.cfg["num_embedding_params"] = num_params(self.embedder)
        conf.cfg["num_total_params"] += conf.cfg["num_embedding_params"]

    def create_graph(self, example):
        return graph_utils.create_graph(example, glove_embedder=self.embedder)



