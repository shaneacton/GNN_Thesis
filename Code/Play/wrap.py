import torch
from torch import nn
from transformers.modeling_longformer import _compute_global_attention_mask as qa_glob_att


class Wrap(nn.Module):

    def __init__(self, pretrained, output):
        super().__init__()
        self.pretrained = pretrained
        self.output = output

        self.middle = nn.Linear(self.pretrained.config.hidden_size, self.output.config.hidden_size)

    def forward(self, input_ids, attention_mask, start_positions, end_positions, return_dict=True):
        # gives global attention to all question tokens
        global_attention_mask = qa_glob_att(input_ids, self.output.config.sep_token_id)

        with torch.no_grad():  # no finetuning the embedder
            embs = self.pretrained(input_ids=input_ids, attention_mask=attention_mask, return_dict=True,
                                   global_attention_mask=global_attention_mask)
            embs = embs["last_hidden_state"]

        embs = self.middle(embs)
        out = self.output(inputs_embeds=embs, attention_mask=attention_mask, return_dict=return_dict,
                          start_positions=start_positions, end_positions=end_positions,
                          global_attention_mask=global_attention_mask)
        return out