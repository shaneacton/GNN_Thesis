from typing import Union

import torch
from torch import nn

from Code.Data.Graph.Embedders.graph_encoding import GraphEncoding
from Code.Training import device


class TypeEmbedder(nn.Module):

    def __init__(self, num_features, graph_feature_type=None):
        """
        :param graph_feature_type: graph_embedding_config.NODE_TYPES, graph_embedding_config.EDGE_TYPES, None
        """
        super().__init__()
        self.graph_feature_type = graph_feature_type
        self.num_features = num_features
        self.embeddings = None

        self.extra_types_tollerance = 5  # maximum extra types expected beyond in the sample data point provided

    def init_embeddings(self, types: torch.Tensor):
        max_id, _ = torch.max(types, dim=0)
        max_id = max_id.item()

        num_emb = max_id + self.extra_types_tollerance
        self.embeddings = nn.Embedding(num_emb, self.num_features).to(device)

    def forward(self, item: Union[GraphEncoding, torch.Tensor]):
        if isinstance(item, GraphEncoding):
            types = item.types.__dict__[self.graph_feature_type]
        else:  # is a types tensor already
            types = item

        if not self.embeddings:
            self.init_embeddings(types)

        return self.embeddings(types)