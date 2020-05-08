from typing import List

import torch
from torch import Tensor

from Code.GNN_Playground.Models import embedded_size
from Code.GNN_Playground.Training import device
from Datasets.Batching.batch_item import BatchItem


class Batch:

    """
        a batch is a collection of data_examples with
        utils to pad and combine vecs from these examples
    """

    def __init__(self):
        self.batch_items: List[BatchItem] = []

    def add_batch_item(self, batch_item: BatchItem):
        self.batch_items.append(batch_item)

    def __len__(self):
        return len(self.batch_items)

    def get_candidates_vec(self):
        return Batch.pad_and_combine([bi.question.get_candidates_embedding() for bi in self.batch_items])

    def get_contexts_vec(self):
        return Batch.pad_and_combine([bi.data_example.context.get_context_embedding() for bi in self.batch_items])

    def get_queries_vec(self):
        return Batch.pad_and_combine([bi.question.get_embedding() for bi in self.batch_items])

    def get_answer_cand_index_vec(self):
        return torch.cat([bi.question.get_answer_cand_index_vec() for bi in self.batch_items], dim=0)

    def get_cqc_vecs(self):
        return self.get_contexts_vec(), self.get_queries_vec(), self.get_candidates_vec()

    def get_answer_type(self):
        return self.batch_items[0].question.get_answer_type()

    @staticmethod
    def pad_and_combine(vecs: List[Tensor], dim=1):
        """
        pads dim to equal size, concats along batch dim
        :param dim: the padding dim which contains differing lengths
        :return: batched seqs
        """
        longest_seq = max([vec.size(dim) for vec in vecs])
        feature_size = vecs[0].size(2)
        batch_size = vecs[0].size(0)

        pad = lambda vec: torch.cat(
            [vec, torch.zeros(batch_size, longest_seq - vec.size(dim), feature_size).to(device)], dim=dim) \
            if vec.size(dim) < longest_seq else vec

        vecs = [pad(vec) for vec in vecs]
        return torch.cat(vecs, dim=0)

