from typing import Tuple

import torch
from torch import nn, Tensor
from torch_geometric.nn import GATConv, SAGEConv
from transformers.modeling_longformer import LongformerPreTrainedModel, \
    LongformerModel, create_position_ids_from_input_ids

# from Code.Play.initialiser import ATTENTION_WINDOW
from Code.Play.gat import Gat
from Code.Training import device

MAX_NODES = 3500


class GatWrap(LongformerPreTrainedModel):

    def __init__(self, pretrained: LongformerModel, output):
        super().__init__(pretrained.config)
        self.pretrained = pretrained
        self.output = output
        self.max_pretrained_pos_ids = pretrained.config.max_position_embeddings
        print("max pos embs:", self.max_pretrained_pos_ids)
        # self.pos_embed = pretrained.embeddings.position_embeddings
        # self.pos_embed_map = nn.Linear(self.pretrained_size, self.middle_size)

        self.middle1 = Gat(self.pretrained_size, self.middle_size, num_edge_types=2)
        self.act = nn.ReLU()
        self.middle2 = Gat(self.middle_size, self.middle_size, num_edge_types=2)

    @property
    def pretrained_size(self):
        return self.pretrained.config.hidden_size

    @property
    def middle_size(self):
        return self.output.config.hidden_size

    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None, return_dict=True):
        # gives global attention to all question and/or candidate tokens
        if input_ids.size(1) >= self.max_pretrained_pos_ids or input_ids.size(1) >= MAX_NODES:
            return torch.tensor([0.] * input_ids.size(0), requires_grad=True)  # too large to pass
        # global_attention_mask = qa_glob_att(input_ids, self.output.config.sep_token_id, before_sep_token=False)
        global_attention_mask = self.get_glob_att_mask(input_ids)
        # print("glob att mask:", global_attention_mask.size(), global_attention_mask)
        pos_ids = self.get_safe_pos_ids(input_ids)
        with torch.no_grad():  # no finetuning the embedder
            embs = self.pretrained(input_ids=input_ids, attention_mask=attention_mask, return_dict=True,
                                   global_attention_mask=global_attention_mask, position_ids=pos_ids)
            embs = embs["last_hidden_state"]

        # print("token embs:", embs.size())
        edge, edge_types = self.get_edge_indices(embs.size(1), global_attention_mask)
        # pos_embs = self.get_pos_embs(input_ids)
        # print("pos embs:", pos_embs.size())

        embs = self.act(self.middle1(x=embs.squeeze(), edge_index=edge, edge_types=edge_types))
        embs = self.act(self.middle2(x=embs, edge_index=edge, edge_types=edge_types)).view(1, -1, self.middle_size)
        # print("after gat:", embs.size())
        out = self.output(inputs_embeds=embs, attention_mask=attention_mask, return_dict=return_dict,
                          start_positions=start_positions, end_positions=end_positions,
                          global_attention_mask=global_attention_mask)
        # print("loss:", out["loss"])
        return out

    def get_safe_pos_ids(self, input_ids: Tensor):
        """
            incase there are more tokens than the max pos ids in the pretrained
            simple wrap performed in case of spill over
        """
        num_ids = input_ids.size(1)
        batch_size = input_ids.size(0)
        if num_ids < self.max_pretrained_pos_ids:
            return None  # is safe
        safe_ids = [i % self.max_pretrained_pos_ids for i in range(num_ids)]
        safe_ids = [safe_ids for _ in range(batch_size)]
        safe_ids = torch.tensor(safe_ids).to(device)
        # print("safe ids:", safe_ids.size())
        return safe_ids

    def get_glob_att_mask(self, input_ids: Tensor) -> Tensor:
        """
            input ids are encoded via <context>sep<query>sep[all candidates]
            thus all ids after the first sep should be global
            adapted from modeling_longformer._compute_global_attention_mask from Transformers lib
        """
        sep_token_indices = (input_ids == self.output.config.sep_token_id).nonzero()
        first_sep = sep_token_indices.squeeze()[0][1].item()
        # print("first:", first_sep)
        # print("sep idxs:", sep_token_indices)
        attention_mask = torch.arange(input_ids.shape[1], device=input_ids.device)
        attention_mask = (attention_mask.expand_as(input_ids) > (first_sep + 1)).to(torch.uint8) * (
                attention_mask.expand_as(input_ids) < input_ids.shape[-1]
        ).to(torch.uint8)
        # print("att mask:", attention_mask)
        return attention_mask

    def get_pos_embs(self, input_ids: Tensor):
        pas_id = self.pretrained.config.pad_token_id
        pos_ids = create_position_ids_from_input_ids(input_ids, pas_id).to(input_ids.device)
        pos_embs = self.pos_embed(pos_ids)
        num_elements, num_features = pos_embs.size(1), pos_embs.size(2)
        """reduce the dim of the pos embs to the same as the GNN"""
        pos_embs = self.pos_embed_map(pos_embs.view(num_elements, num_features)).view(1, num_elements, self.middle_size)
        return pos_embs

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

