import torch
from torch import nn
from torch_geometric.nn import GATConv
from transformers.modeling_longformer import _compute_global_attention_mask as qa_glob_att, LongformerPreTrainedModel

# from Code.Play.initialiser import ATTENTION_WINDOW
from Code.Training import device


class GatWrap(LongformerPreTrainedModel):

    def __init__(self, pretrained, output):
        super().__init__(pretrained.config)
        self.pretrained = pretrained
        self.output = output

        # self.middle = nn.Linear(pretrained.config.hidden_size, output.config.hidden_size)
        self.middle1 = GATConv(self.pretrained_size, self.middle_size)
        self.act = nn.ReLU()
        self.middle2 = GATConv(self.middle_size, self.middle_size)


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

        embs = self.act(self.middle1(x=embs.squeeze(), edge_index=edge))
        embs = self.act(self.middle2(x=embs, edge_index=edge).view(1, -1, self.middle_size))
        # print("after gat:", embs.size())
        out = self.output(inputs_embeds=embs, attention_mask=attention_mask, return_dict=return_dict,
                          start_positions=start_positions, end_positions=end_positions,
                          global_attention_mask=global_attention_mask)
        return out

    @staticmethod
    def get_edge_indices(num_tokens, glob_att_mask: torch.Tensor, window_size=128) -> torch.Tensor:
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

        return torch.tensor([froms, tos]).to(device)

