import torch
from torch import nn

from Code.Data.Graph.Embedders.sequence_summariser import SequenceSummariser
from Code.Training import device


class HeadAndTailCat(SequenceSummariser):

    def __init__(self):
        super().__init__()
        self.feature_reducer: nn.Module = None

    def _init_layers(self, feature_size):
        self.feature_reducer = nn.Linear(feature_size * 2, feature_size).to(device=device)

    def _summarise(self, embedded_sequence: torch.Tensor):
        batch_size = embedded_sequence.size(0)

        cat_seq = torch.cat([embedded_sequence[:,0,:], embedded_sequence[:,-1,:]], dim=1)
        return self.feature_reducer(cat_seq).view(batch_size, 1, -1)
