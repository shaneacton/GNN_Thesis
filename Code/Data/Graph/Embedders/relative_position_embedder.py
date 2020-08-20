from typing import Dict, List

import torch
from torch import nn

from Code.Config import GraphEmbeddingConfig, GraphConstructionConfig
from Code.Config import graph_construction_config as construction
from Code.Data.Graph.Nodes.node_position import NodePosition, incompatible


class RelativePositionEmbedder(nn.Module):

    def __init__(self, num_features, gcc: GraphConstructionConfig, gec: GraphEmbeddingConfig):
        super().__init__()
        self.gcc = gcc
        self.gec = gec
        self.num_features = num_features

        self.pos_to_id_map: Dict[NodePosition, int] = {incompatible: 0}
        self.next_id = 1

        self.embeddings = None
        self.init_embeddings()

    def init_embeddings(self):
        context_levels = self.gcc.context_structure_nodes
        query_levels = [x.split("query_")[1] for x in self.gcc.query_structure_nodes]

        sources = [construction.CONTEXT] * len(context_levels) + [construction.QUERY] * len(query_levels)
        levels = context_levels + query_levels

        for l in range(len(levels)):
            # registers all node positions anticipated
            level = levels[l]
            source = sources[l]

            window_size = self.gec.relative_embeddings_window_per_level[level]
            for i in range(-window_size, window_size):
                position = NodePosition(source, level, i, window_size)
                self.pos_to_id_map[position] = self.next_id
                self.next_id += 1

        self.embeddings = nn.Embedding(self.next_id, self.num_features)

    def forward(self, relative_positions: List[NodePosition]):
        position_ids = torch.tensor([self.pos_to_id_map[pos] for pos in relative_positions])
        return self.embeddings(position_ids)
