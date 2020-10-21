from typing import List, Dict

import torch
from torch import nn

import Code.constants
from Code.Config import graph_construction_config as construction
from Code.Data.Graph.Nodes.node_position import NodePosition, incompatible
from Code.Training import device


class PositionEmbedder(nn.Module):

    def __init__(self, num_features, gcc, gec):
        super().__init__()
        self.gcc = gcc
        self.gec = gec
        self.num_features = num_features

        self.pos_to_id_map: Dict[NodePosition, int] = {incompatible: 0}
        self.next_id = 1

        self.embeddings = None
        self.init_embeddings()

    def forward(self, positions: List[NodePosition]):
        positions = [pos if pos else incompatible for pos in positions]
        try:
            position_ids = torch.tensor([self.pos_to_id_map[pos] for pos in positions]).to(device)
        except Exception as ex:
            # print("map:",self.pos_to_id_map)
            for pos in positions:
                if pos in self.pos_to_id_map:
                    continue
                print("pos:", pos, "in:", (pos in self.pos_to_id_map))
            raise ex
        return self.embeddings(position_ids)

    def init_embeddings(self):
        """
        trains a separate embedding for relative positions at different levels, eg sent2sent or token2token
        loops through every possible relative position input to get total number of embeddings needed
        """
        context_levels = self.gcc.context_structure_levels
        # doc nodes don't have positional encodings
        context_levels = [lev for lev in context_levels if lev != Code.constants.DOCUMENT]
        query_levels = [x.split("query_")[1] for x in self.gcc.query_structure_levels]

        sources = [Code.constants.CONTEXT] * len(context_levels) + [Code.constants.QUERY] * len(query_levels)
        levels = context_levels + query_levels

        # print("levels:", levels, "sources:",sources)

        for l in range(len(levels)):
            # registers all node positions anticipated
            level = levels[l]
            source = sources[l]

            # print("level:", level, "source:",source, "window_size:",window_size)
            window_size = self.gec.relative_embeddings_window_per_level[level]

            for i in self.get_expected_position_ids(level):
                position = NodePosition(source, level, i, window_size)
                self.pos_to_id_map[position] = self.next_id
                self.next_id += 1

        self.embeddings = nn.Embedding(self.next_id, self.num_features)

    def get_expected_position_ids(self, level):
        return range(self.gec.num_positional_embeddings)