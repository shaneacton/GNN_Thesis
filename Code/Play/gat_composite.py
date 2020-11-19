import torch
from torch import nn, Tensor
from torch_geometric.nn import GATConv, SAGEConv
from transformers.modeling_longformer import _compute_global_attention_mask as qa_glob_att, LongformerPreTrainedModel, \
    LongformerModel, create_position_ids_from_input_ids

# from Code.Play.initialiser import ATTENTION_WINDOW
from Code.Play.Gat import Gat
from Code.Training import device


class GatWrap(LongformerPreTrainedModel):

    def __init__(self, pretrained: LongformerModel, output):
        super().__init__(pretrained.config)
        self.pretrained = pretrained
        self.pos_embed = pretrained.embeddings.position_embeddings
        self.output = output

        # self.middle = nn.Linear(pretrained.config.hidden_size, output.config.hidden_size)
        self.middle1 = Gat(self.pretrained_size, self.middle_size)
        self.act = nn.ReLU()
        self.middle2 = Gat(self.middle_size, self.middle_size)

    @property
    def pretrained_size(self):
        return self.pretrained.config.hidden_size

    @property
    def middle_size(self):
        return self.output.config.hidden_size

    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None, return_dict=True):
        # gives global attention to all question tokens
        global_attention_mask = qa_glob_att(input_ids, self.output.config.sep_token_id)
        # print("glob att mask:", global_attention_mask.size(), global_attention_mask)
        with torch.no_grad():  # no finetuning the embedder
            embs = self.pretrained(input_ids=input_ids, attention_mask=attention_mask, return_dict=True,
                                   global_attention_mask=global_attention_mask)
            embs = embs["last_hidden_state"]

        # print("token embs:", embs.size())
        edge = self.get_edge_indices(embs.size(1), global_attention_mask).squeeze()
        # pos_embs = self.get_pos_embs(input_ids)
        # print("pos embs:", pos_embs.size())

        embs = self.act(self.middle1(x=embs.squeeze(), edge_index=edge))
        embs = self.act(self.middle2(x=embs, edge_index=edge)).view(1, -1, self.middle_size)
        # print("after gat:", embs.size())
        out = self.output(inputs_embeds=embs, attention_mask=attention_mask, return_dict=return_dict,
                          start_positions=start_positions, end_positions=end_positions,
                          global_attention_mask=global_attention_mask)
        return out

    def get_pos_embs(self, input_ids: Tensor):
        pas_id = self.pretrained.config.pad_token_id
        pos_ids = create_position_ids_from_input_ids(input_ids, pas_id).to(input_ids.device)
        pos_embs = self.pos_embed(pos_ids)
        return pos_embs

    @staticmethod
    def get_edge_indices(num_tokens, glob_att_mask: Tensor, window_size=128) -> Tensor:
        # print("num tokens:", num_tokens, type(num_tokens))
        froms = []
        tos = []
        for from_id in range(num_tokens - 1):
            """sliding window connections"""
            if glob_att_mask[0, from_id].item() == 1:
                "global, so all"
                # print("global on token", from_id)
                for to_id in range(num_tokens - 1):
                    if to_id == from_id:
                        continue
                    froms.append(from_id)
                    tos.append(to_id)
            else:
                """not global, so sliding"""
                for d in range(1, window_size):
                    to_id = from_id + d
                    if to_id >= num_tokens:
                        break
                    froms.append(from_id)
                    tos.append(to_id)

        # print("num edges:", len(froms))
        edges = torch.tensor([froms, tos]).to(device)
        # print("edges:", edges.size(), edges)
        return edges

