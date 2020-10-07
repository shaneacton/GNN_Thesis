from typing import List

import torch
from torch import Tensor

from Code.Data.Text.Answers.candidate_answer import CandidateAnswer
from Code.Data.Text.Answers.extracted_answer import ExtractedAnswer
from Code.Training import device
from Datasets.Batching.batch_item import BatchItem


class Batch:

    """
        a batch is a collection of data_examples with
        utils to pad and combine vecs from these examples
    """

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.batch_items: List[BatchItem] = []

    def add_batch_item(self, batch_item: BatchItem):
        self.batch_items.append(batch_item)

    def __len__(self):
        return len(self.batch_items)

    def get_answer_cand_index_tensor(self):
        """dim ~ (batch)"""
        return torch.cat([bi.question.get_answer_cand_index_vec() for bi in self.batch_items], dim=0)

    def get_answer_type(self):
        return self.batch_items[0].question.get_answer_type()

    def get_answers_tensor(self) -> Tensor:
        if self.get_answer_type() == CandidateAnswer:
            return self.get_answer_cand_index_tensor()
        if self.get_answer_type() == ExtractedAnswer:
            return self.get_answer_span_vec()

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

    def __repr__(self):
        return "Batch("+"\n\n".join([repr(bi) for bi in self.batch_items])

    def get_answer_span_vec(self):
        """spans are shape: (batch, 2)"""
        spans = [bi.data_sample.get_answer_span_vec(bi.question.answers) for bi in self.batch_items]
        return torch.cat(spans, dim=0)


