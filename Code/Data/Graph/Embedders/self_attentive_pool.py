from typing import List

import torch

from Code.Data.Graph.Embedders.sequence_summariser import SequenceSummariser


class SelfAttentivePool(SequenceSummariser):
    def summarise(self, embedded_sequence: List[torch.Tensor]):
        return None