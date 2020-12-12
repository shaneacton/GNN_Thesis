from typing import Tuple

import torch
from torch import nn, Tensor
from torch_geometric.nn import GATConv
from transformers.modeling_longformer import LongformerPreTrainedModel

from transformers.modeling_outputs import QuestionAnsweringModelOutput

from Code.Data.Text.longformer_embedder import LongformerEmbedder
from Code.Play.gat import Gat
from Code.Training import device

MAX_NODES = 2900  # 2900


class GatWrapLongEnc(LongformerPreTrainedModel):

    def __init__(self, _, output):
        self.middle_size = output.config.hidden_size
        long_embedder = LongformerEmbedder(out_features=self.middle_size)
        super().__init__(long_embedder.longformer.config)
        self.long_embedder = long_embedder
        self.middle1 = GATConv(self.middle_size, self.middle_size)
        self.act = nn.ReLU()
        self.middle2 = GATConv(self.middle_size, self.middle_size)

        self.output = output
        self.max_pretrained_pos_ids = self.long_embedder.longformer.config.max_position_embeddings
        print("max pos embs:", self.max_pretrained_pos_ids)

    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None, return_dict=True):
        # gives global attention to all question and/or candidate tokens
        if input_ids.size(1) >= self.max_pretrained_pos_ids or input_ids.size(1) >= MAX_NODES:  # too large to pass
            has_loss = start_positions is not None and end_positions is not None
            return self.get_null_return(input_ids, return_dict, has_loss)
        embs = self.long_embedder.embed(input_ids, attention_mask)

        # print("token embs:", embs.size())
        global_attention_mask = self.long_embedder.get_glob_att_mask(input_ids)
        edge, edge_types = self.get_edge_indices(embs.size(1), global_attention_mask)

        embs = self.act(self.middle1(x=embs.squeeze(), edge_index=edge, edge_types=edge_types))
        embs = self.act(self.middle2(x=embs, edge_index=edge, edge_types=edge_types)).view(1, -1, self.middle_size)
        # print("after gat:", embs.size())
        out = self.output(inputs_embeds=embs, attention_mask=attention_mask, return_dict=return_dict,
                          start_positions=start_positions, end_positions=end_positions,
                          global_attention_mask=global_attention_mask)
        return out

    @staticmethod
    def get_edge_indices(num_tokens, glob_att_mask: Tensor, window_size=128) -> Tuple[Tensor, Tensor]:
        # print("num tokens:", num_tokens, type(num_tokens))
        froms = []
        tos = []

        edge_types = []
        for from_id in range(num_tokens):
            """sliding window connections"""
            if glob_att_mask[0, from_id].item() == 1:
                "global, so all"
                # print("global on token", from_id)
                for to_id in range(num_tokens):
                    # eff edges:63153, calc edges:62641
                    # if to_id == from_id:
                    #     continue
                    froms.append(from_id)
                    tos.append(to_id)
                    edge_types.append(0)
            else:
                """not global, so sliding"""
                for d in range(0, window_size):
                    to_id = from_id + d
                    if to_id >= num_tokens:
                        break
                    froms.append(from_id)
                    tos.append(to_id)
                    edge_types.append(1)

        # print("num edges:", len(froms))
        edges = torch.tensor([froms, tos]).to(device)
        edge_types = torch.tensor(edge_types).to(device)
        # print("edges:", edges.size(), edges)
        return edges, edge_types

    def get_null_return(self, input_ids:Tensor, return_dict:bool, include_loss:bool):
        loss = None
        num_ids = input_ids.size(1)
        if include_loss:
            loss = torch.tensor(0., requires_grad=True)
        logits = torch.tensor([0.] * num_ids, requires_grad=True).to(float)

        if not return_dict:
            output = (logits, logits)
            return ((loss,) + output) if loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=logits,
            end_logits=logits,
        )
