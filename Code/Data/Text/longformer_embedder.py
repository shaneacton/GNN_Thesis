import torch
from torch import nn, Tensor
from transformers import LongformerModel, BatchEncoding

from Code.Data.Text.embedder import Embedder
from Code.Training import device
from Code.Training.Utils.initialiser import get_pretrained_longformer


class LongformerEmbedder(Embedder):
    """a convenience wrapper around the Longformer which returns the last hidden states"""

    def __init__(self, longformer: LongformerModel=None, out_features=-1, num_types=3):
        super().__init__()
        if not longformer:
            longformer = get_pretrained_longformer()
        for param in longformer.parameters():
            param.requires_grad = False

        longformer.embeddings.token_type_embeddings = nn.Embedding(num_types, longformer.config.hidden_size)

        self.longformer: LongformerModel = longformer.to(device)
        self.feature_mapper = None
        self.out_features = longformer.config.hidden_size
        if out_features != longformer.config.hidden_size and out_features != -1:
            self.out_features = out_features
            self.feature_mapper = nn.Linear(longformer.config.hidden_size, out_features)

    def embed(self, input_ids, attention_mask, all_global=False):
        position_ids = self.get_safe_pos_ids(input_ids)
        if len(input_ids.size()) == 1:
            #  batch size is 1
            input_ids = input_ids.view(1, -1)
            attention_mask = attention_mask.view(1, -1)

        if all_global:
            global_attention_mask = torch.ones(input_ids.size()).to(device)
        else:
            """we use <context><query>[cands] so global att is after sep, not before"""
            global_attention_mask = self.get_glob_att_mask(input_ids).to(device)
        with torch.no_grad():  # no finetuning the embedder
            embs = self.longformer(input_ids=input_ids, attention_mask=attention_mask, return_dict=True,
                                   global_attention_mask=global_attention_mask, position_ids=position_ids)
            embs = embs["last_hidden_state"]

        if self.feature_mapper:
            embs = self.feature_mapper(embs)
        return embs

    def get_safe_pos_ids(self, input_ids: Tensor) -> Tensor:
        max_ids = self.longformer.config.max_position_embeddings
        num_toks = input_ids.size(-1)
        # print("num toks:", num_toks, "/", max_ids)
        if num_toks < max_ids - 1:
            return None  # auto generated will be safe
        """too many input ids for longformer. must wrap"""
        return torch.tensor([i % max_ids for i in range(num_toks)]).to(device)

    def get_first_sep_index(self, input_ids: Tensor):
        sep_token_indices = (input_ids == self.longformer.config.sep_token_id).nonzero()
        first_sep = sep_token_indices.squeeze()[0][1].item()
        return first_sep

    def get_glob_att_mask(self, input_ids: Tensor) -> Tensor:
        """
            input ids are encoded via <context>sep<query>sep[all candidates]
            thus all ids after the first sep should be global
            adapted from modeling_longformer._compute_global_attention_mask from Transformers lib
        """
        first_sep = self.get_first_sep_index(input_ids)
        num_ids = input_ids.shape[1]
        return self.get_glob_att_mask_from(first_sep, num_ids)

    @staticmethod
    def get_glob_att_mask_from(glob_start_id: int, length: int, batch_size=1) -> Tensor:
        """returns a glob att mask with globs after the glob_start_id"""
        attention_mask = torch.arange(length, device=device)
        tens = torch.zeros(batch_size, length)
        attention_mask = (attention_mask.expand_as(tens) > (glob_start_id + 1)).to(torch.uint8) * (
                attention_mask.expand_as(tens) < length
        ).to(torch.uint8)
        # print("att mask:", attention_mask)
        return attention_mask
