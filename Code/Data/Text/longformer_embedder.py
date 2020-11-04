import torch
from torch import nn
from transformers import LongformerModel, BatchEncoding
from transformers.modeling_longformer import _compute_global_attention_mask as qa_glob_att


from Code.Data.Text.embedder import Embedder


class LongformerEmbedder(Embedder):
    """a convenience wrapper around the Longformer which returns the last hidden states"""

    def __init__(self, longformer: LongformerModel=None, out_features=-1):
        super().__init__()
        if not longformer:
            longformer = LongformerModel.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")
        self.longformer = longformer
        self.feature_mapper = None
        if out_features != -1:
            self.feature_mapper = nn.Linear(longformer.config.hidden_size)

    def embed(self, encoding: BatchEncoding):
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        global_attention_mask = qa_glob_att(input_ids, self.output.config.sep_token_id)
        with torch.no_grad():  # no finetuning the embedder
            embs = self.pretrained(input_ids=input_ids, attention_mask=attention_mask, return_dict=True,
                                   global_attention_mask=global_attention_mask)
            embs = embs["last_hidden_state"]

        if self.feature_mapper:
            embs = self.feature_mapper(embs)
        return embs
