import torch
from torch import nn, Tensor
from transformers import LongformerModel, BatchEncoding

from Code.Data.Text.embedder import Embedder
from Code.Training import device


class LongformerEmbedder(Embedder):
    """a convenience wrapper around the Longformer which returns the last hidden states"""

    def __init__(self, longformer: LongformerModel=None, out_features=-1):
        super().__init__()
        if not longformer:
            longformer = LongformerModel.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")

        self.longformer = longformer.to(device)
        self.feature_mapper = None
        if out_features != longformer.config.hidden_size and out_features != -1:
            self.feature_mapper = nn.Linear(longformer.config.hidden_size, out_features)

    def embed(self, encoding: BatchEncoding):
        input_ids = Tensor(encoding["input_ids"]).type(torch.LongTensor).to(device)
        attention_mask = Tensor(encoding["attention_mask"]).type(torch.LongTensor).to(device)
        if len(input_ids.size()) == 1:
            #  batch size is 1
            input_ids = input_ids.view(1, -1)
            attention_mask = attention_mask.view(1, -1)

        """we use <context><query>[cands] so global att is after sep, not before"""
        global_attention_mask = self.get_glob_att_mask(input_ids).to(device)
        with torch.no_grad():  # no finetuning the embedder
            embs = self.longformer(input_ids=input_ids, attention_mask=attention_mask, return_dict=True,
                                   global_attention_mask=global_attention_mask)
            embs = embs["last_hidden_state"]

        if self.feature_mapper:
            embs = self.feature_mapper(embs)
        return embs

    def get_glob_att_mask(self, input_ids: Tensor) -> Tensor:
        """
            input ids are encoded via <context>sep<query>sep[all candidates]
            thus all ids after the first sep should be global
            adapted from modeling_longformer._compute_global_attention_mask from Transformers lib
        """
        sep_token_indices = (input_ids == self.longformer.config.sep_token_id).nonzero()
        first_sep = sep_token_indices.squeeze()[0][1].item()
        # print("first:", first_sep)
        # print("sep idxs:", sep_token_indices)
        attention_mask = torch.arange(input_ids.shape[1], device=input_ids.device)
        attention_mask = (attention_mask.expand_as(input_ids) > (first_sep + 1)).to(torch.uint8) * (
                attention_mask.expand_as(input_ids) < input_ids.shape[-1]
        ).to(torch.uint8)
        # print("att mask:", attention_mask)
        return attention_mask