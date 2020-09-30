from typing import Iterator

from Code.Data.Graph.Datasets.graph_dataset import GraphDataset
from Code.Data.Graph.context_graph import ContextGraph


class OnlineGraphDataset(GraphDataset):

    def get_context_graphs(self) -> Iterator[ContextGraph]:
        for batch in self.batch_text_reader.get_all_batches():
            for batch_item in batch.batch_items:
                graph = self.graph_constructor.create_graph_from_data_sample(batch_item.data_sample)
                yield graph

    def process(self):
        pass
