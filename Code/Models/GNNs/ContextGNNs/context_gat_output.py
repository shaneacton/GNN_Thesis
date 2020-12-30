from typing import Dict

import torch
from torch import nn
from transformers import LongformerForQuestionAnswering

from Code.Config import GNNConfig
from Code.Data.Graph.Embedders.graph_embedder import GraphEmbedder
from Code.Data.Graph.Nodes.token_node import TokenNode
from Code.Data.Graph.graph_encoding import GraphEncoding
from Code.Models.GNNs.ContextGNNs.context_gat import ContextGAT
from Code.Models.GNNs.OutputModules.span_selection import SpanSelection
from Code.Models.loss_funcs import get_span_loss
from Code.Training import device
from Code.Training.Utils.initialiser import get_fresh_span_longformer


class ContextGATOutput(ContextGAT):

    def __init__(self, graph_embedder: GraphEmbedder, gnnc: GNNConfig, in_features):
        super().__init__(graph_embedder, gnnc)
        self.in_features = in_features
        self.qa_outputs = nn.Linear(in_features, 2)

    def pass_through_output(self, data: GraphEncoding, **kwargs):
        start_positions = kwargs.pop("start_positions", None)
        end_positions = kwargs.pop("end_positions", None)
        node_embs = torch.cat([torch.zeros(1, self.in_features).to(device), data.x], dim=0).view(1, -1, self.in_features)

        logits = self.qa_outputs(node_embs)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits, end_logits = start_logits.squeeze().view(1, -1), end_logits.squeeze().view(1, -1)
        if start_positions and end_positions:
            loss = get_span_loss(start_positions, end_positions, start_logits, end_logits)
            return loss, start_logits, end_logits

        return start_logits,  end_logits

    def init_output_model(self, example: Dict, in_features):
        """this version creates its output layer in the init"""
        pass