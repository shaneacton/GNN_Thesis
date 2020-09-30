import os
import pickle
from typing import Iterator

from Code.Config import configs
from Code.Data.Graph.Datasets.graph_dataset import GraphDataset
from Code.Data.Graph.context_graph import ContextGraph
from Viz.graph_visualiser import render_graph


class PreprocessedGraphDataset(GraphDataset):

    def get_context_graphs(self) -> Iterator[ContextGraph]:
        i = 0

        for batch in self.batch_text_reader.get_all_batches():
            for _ in batch.batch_items:
                file_path = self.get_data_file_path(i)
                filehandler = open(file_path, 'rb')

                graph = pickle.load(filehandler)
                filehandler.close()
                i+=1
                yield graph

    def process(self):
        """save the context graphs if this construction variant has not been preprocessed yet"""
        if self.previously_processed():
            # no need to reparse this dataset if it has already been saved
            return

        i = 0
        for batch in self.batch_text_reader.get_all_batches():
            for batch_item in batch.batch_items:
                file_path = self.get_data_file_path(i)
                filehandler = open(file_path, 'wb')

                try:
                    graph = self.graph_constructor.create_graph_from_data_sample(batch_item.data_sample)
                except:
                    """cannot convert to graph"""
                    continue

                pickle.dump(graph, filehandler)
                filehandler.close()

                print("saving graph to", file_path)

                i += 1

        self.write_meta(length=i)

    def previously_processed(self):
        """checks if this (dataset,variant) tuple has already been preprocessed"""
        return os.path.exists(self.get_meta_path())


if __name__ == "__main__":
    from Datasets.Batching.batch_reader import BatchReader
    from Datasets.Readers.qangaroo_reader import QUangarooDatasetReader
    from Datasets.Readers.squad_reader import SQuADDatasetReader

    squad_reader = SQuADDatasetReader("SQuAD")
    qangaroo_reader = QUangarooDatasetReader("wikihop")

    wikihop_path = QUangarooDatasetReader.dev_set_location("wikihop")
    squad_path = SQuADDatasetReader.dev_set_location()

    qangaroo_batch_reader = BatchReader(qangaroo_reader, 1, wikihop_path)
    squad_batch_reader = BatchReader(squad_reader, 1, squad_path)

    reader = qangaroo_batch_reader

    pgd = PreprocessedGraphDataset(reader, configs.get_graph_constructor())

    for i, graph in enumerate(pgd.get_context_graphs()):
        if i>5:
            break

        render_graph(graph, "test" + repr(i), reader.data_reader.datset_name)

