from typing import List

import torch

from Code.Data.Graph.Embedders.sequence_summariser import SequenceSummariser


class HeadAndTailCat(SequenceSummariser):

    def summarise(self, embedded_sequence: List[torch.Tensor]):
        return torch.cat([embedded_sequence[0], embedded_sequence[-1]], dim=0)
