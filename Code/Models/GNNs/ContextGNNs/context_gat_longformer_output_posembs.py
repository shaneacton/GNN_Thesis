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


class ContextGATOutputPos(ContextGAT):

    def __init__(self, graph_embedder: GraphEmbedder, gnnc: GNNConfig, in_features):
        super().__init__(graph_embedder, gnnc)
        self.in_features = in_features
        self.qa_outputs = nn.Linear(in_features, 2)

    @property
    def pos_embedder(self):
        return self.graph_embedder.long_embedder.longformer.embeddings.position_embeddings

    def _forward(self, data: GraphEncoding, **kwargs):
        """add pos embs and pass on"""
        ids = torch.tensor([i for i in range(data.num_nodes)]).to(device)
        safe_ids = self.graph_embedder.long_embedder.get_safe_pos_ids(ids)
        ids = safe_ids if safe_ids else ids
        pos_embs = self.pos_embedder(ids)
        # print("x:", data.x.size(), "pos:", pos_embs.size())
        data.x = data.x + pos_embs
        return super(ContextGATOutputPos, self)._forward(data, **kwargs)

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
        """makes pos embedder fine tunable"""
        for param in self.pos_embedder.parameters():
            param.requires_grad = True