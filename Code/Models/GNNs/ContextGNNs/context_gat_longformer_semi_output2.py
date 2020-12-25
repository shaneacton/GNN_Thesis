from typing import Dict

import torch
from torch import nn
from transformers import LongformerForQuestionAnswering

from Code.Config import GNNConfig
from Code.Data.Graph.Embedders.graph_embedder import GraphEmbedder
from Code.Models.GNNs.ContextGNNs.context_gat import ContextGAT
from Code.Models.GNNs.OutputModules.span_selection import SpanSelection
from Code.Models.loss_funcs import get_span_loss
from Code.Training import device
from Code.Training.Utils.initialiser import get_fresh_span_longformer


class ContextGATLongSemiOutput2(ContextGAT):

    def __init__(self, graph_embedder: GraphEmbedder, gnnc: GNNConfig, in_features):
        super().__init__(graph_embedder, gnnc)
        self.output_model: LongformerForQuestionAnswering = get_fresh_span_longformer(in_features)
        self.in_features = in_features
        self.qa_outputs = nn.Linear(in_features, 2)

    def pass_through_output(self, data, **kwargs):
        start_positions = kwargs.pop("start_positions", None)
        end_positions = kwargs.pop("end_positions", None)
        x_shape = data.x.size()
        node_embs = torch.cat([torch.zeros(1, self.in_features).to(device), data.x], dim=0).view(1, -1, self.in_features)
        num_nodes = node_embs.shape[1]
        global_attention_mask = self.graph_embedder.long_embedder.get_glob_att_mask_from(round(num_nodes * 0.8), num_nodes)

        out = self.output_model(inputs_embeds=node_embs, return_dict=True,
                          start_positions=start_positions.to(device), end_positions=end_positions.to(device),
                          global_attention_mask=global_attention_mask, output_hidden_states=True)

        embs = out["hidden_states"][-1].squeeze()  # last hidden
        embs = embs[1:node_embs.size(1), :]  # cuts off fake sep at start and padding at end
        if embs.size() != x_shape:
            raise Exception()
        data.x = embs.to(device)
        kwargs["start_positions"] = start_positions
        kwargs["end_positions"] = end_positions

        logits = self.qa_outputs(embs)
        start_logits, end_logits = logits.split(1, dim=-1)
        if start_positions and end_positions:
            loss = get_span_loss(kwargs["start_positions"], kwargs["end_positions"], start_logits, end_logits)
            return loss, start_logits, end_logits

        return start_logits,  end_logits

    def init_output_model(self, example: Dict, in_features):
        """this version creates its output layer in the init"""
        pass