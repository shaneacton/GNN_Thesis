import torch
from torch import nn
from torch_geometric.nn import GATConv
from transformers.modeling_longformer import _compute_global_attention_mask as qa_glob_att, LongformerPreTrainedModel

# from Code.Play.initialiser import ATTENTION_WINDOW


class GatWrap(LongformerPreTrainedModel):

    def __init__(self, pretrained, output):
        super().__init__(pretrained.config)
        self.pretrained = pretrained
        self.output = output

        # self.middle = nn.Linear(pretrained.config.hidden_size, output.config.hidden_size)
        self.middle = GATConv(pretrained.config.hidden_size, output.config.hidden_size)

    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None, return_dict=True):
        # gives global attention to all question tokens
        global_attention_mask = qa_glob_att(input_ids, self.output.config.sep_token_id)

        with torch.no_grad():  # no finetuning the embedder
            embs = self.pretrained(input_ids=input_ids, attention_mask=attention_mask, return_dict=True,
                                   global_attention_mask=global_attention_mask)
            embs = embs["last_hidden_state"]

        print("token embs:", embs.size())
        edge = self.get_edge_indices(embs.size(1))
        embs = self.middle(embs)
        out = self.output(inputs_embeds=embs, attention_mask=attention_mask, return_dict=return_dict,
                          start_positions=start_positions, end_positions=end_positions,
                          global_attention_mask=global_attention_mask)
        return out

    def get_edge_indices(self, num_tokens, window_size=ATTENTION_WINDOW):
        print("num tokens:", num_tokens, type(num_tokens))
        for n, node in enumerate(nodes):
            for d in range(1, window_size):
                to_id = n + d
                if to_id >= len(nodes):
                    break
        froms = []