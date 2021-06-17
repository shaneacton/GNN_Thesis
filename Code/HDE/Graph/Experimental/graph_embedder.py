from typing import List

from torch import nn, Tensor

from Code.HDE.hde_model import PadVolumeOverflow
from Code.Transformers.coattention import Coattention
from Code.Transformers.summariser import Summariser
from Code.Transformers.switch_summariser import SwitchSummariser
from Code.Utils.model_utils import num_params
from Config.config import conf


class GraphEmbedder(nn.Module):

    """
        responsible for converting a graph into node features
        controls coattention and summarisers to derive node states given passage, query
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.coattention = Coattention(**kwargs)
        if conf.use_switch_summariser:
            self.summariser = SwitchSummariser(**kwargs)
        else:
            self.summariser = Summariser(**kwargs)

        conf.cfg["num_summariser_params"] = num_params(self.summariser)
        conf.cfg["num_graph_embedder_params"] = num_params(self.summariser) + num_params(self.coattention)

    def get_node_features(self, graph, support_embeddings: List[Tensor], entity_embeddings: List[Tensor], query_embedding: Tensor):
        pass

    def get_query_aware_embeddings(self, target_embeddings: List[Tensor], query_emb: Tensor):
        """uses the coattention module to bring info from the query into the context"""
        # print("supps:", [s.size() for s in support_embeddings])
        pad_volume = max([s.size(1) for s in target_embeddings]) * len(target_embeddings)
        if pad_volume > conf.max_pad_volume:
            raise PadVolumeOverflow()
        if conf.show_memory_usage_data:
            print("documents padded volume:", pad_volume)
        # print("pad vol:", pad_volume)
        query_emb = self.query_contextualiser(query_emb)

        target_embeddings = self.coattention.batched_coattention(target_embeddings, query_emb)
        return target_embeddings