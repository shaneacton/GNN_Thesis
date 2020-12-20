from typing import Dict

import torch

from Code.Config import GNNConfig
from Code.Data.Graph.Embedders.graph_embedder import GraphEmbedder
from Code.Models.GNNs.ContextGNNs.context_gat import ContextGAT
from Code.Training import device
from Code.Training.Utils.initialiser import get_fresh_span_longformer


class ContextGATLongOutput(ContextGAT):

    def __init__(self, graph_embedder: GraphEmbedder, gnnc: GNNConfig, in_features):
        super().__init__(graph_embedder, gnnc)
        self.output_model = get_fresh_span_longformer(in_features)
        self.in_features = in_features

    def pass_through_output(self, data, **kwargs):
        start_positions = kwargs.pop("start_positions", None)
        end_positions = kwargs.pop("end_positions", None)
        # print("start poses:", start_positions)
        node_embs = torch.cat([torch.zeros(1, self.in_features).to(device), data.x], dim=0).view(1, -1, self.in_features)
        num_nodes = node_embs.shape[0]

        global_attention_mask = self.graph_embedder.long_embedder.get_glob_att_mask_from(round(num_nodes * 0.8), num_nodes)

        out = self.output_model(inputs_embeds=node_embs, return_dict=False,
                          start_positions=start_positions.to(device), end_positions=end_positions.to(device),
                          global_attention_mask=global_attention_mask)
        return out

    def init_output_model(self, example: Dict, in_features):
        """this version creates its output layer in the init"""
        pass