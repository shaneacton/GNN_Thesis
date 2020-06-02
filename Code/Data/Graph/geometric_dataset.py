import json
import os
from typing import List

import torch
from torch_geometric.data import Data, Dataset, DataLoader

from Code.Data.Graph.Contructors.compound_graph_constructor import CompoundGraphConstructor
from Code.Data.Graph.Contructors.coreference_constructor import CoreferenceConstructor
from Code.Data.Graph.Contructors.document_node_constructor import DocumentNodeConstructor
from Code.Data.Graph.Contructors.entities_constructor import EntitiesConstructor
from Code.Data.Graph.Contructors.passage_constructor import PassageConstructor
from Code.Data.Graph.Contructors.sentence_contructor import SentenceConstructor
from Code.Data.Graph.Contructors.sequential_entity_linker import SequentialEntityLinker
from Code.Models.GNNs.prop_and_pool import PropAndPool
from Code.Training import device
from Datasets.Batching.batch_reader import BatchReader


def get_geometric_dataset_folder():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..","..","..","Datasets","Geometric")


class GraphDataset(Dataset):

    ROOT = get_geometric_dataset_folder()

    def __init__(self, batch_reader: BatchReader, graph_constructor: CompoundGraphConstructor):
        if batch_reader.batch_size > 1:
            raise Exception("must provide batch reader with size 1 to graph dataset")

        self.batch_reader=batch_reader
        self.graph_constructor=graph_constructor

        super(GraphDataset, self).__init__(self.dataset_path, None, None)
        # process is called by super

    @property
    def dataset_path(self):
        return os.path.join(GraphDataset.ROOT, self.batch_reader.dataset)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data_{}.pt'.format(i) for i in range(len(self))]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def previously_processed(self):
        return os.path.exists(self.get_meta_path())

    def process(self):
        if self.previously_processed():
            # no need to reparse this dataset if it has already been saved
            return
        for i, datapoint in enumerate(self.get_data_objects()):
            torch.save(datapoint, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))

        self.write_meta(i)

    def get_data_objects(self) -> List[Data]:

        for i, batch in enumerate(self.batch_reader.get_batches()):
            for batch_item in batch.batch_items:
                try:
                    graph = self.graph_constructor.append(None, batch_item.data_sample)
                except Exception as e:
                    print(e)
                    continue
                label = batch.get_answers().squeeze()  # chops off batch dim
                graph.set_label(label)
                yield graph.data

    def get_meta_path(self):
        return os.path.join(self.processed_dir,"meta.json")

    def write_meta(self, length):
        meta = {"length": length,"dataset":self.batch_reader.dataset,"graphtype":self.graph_constructor.type}
        with open(self.get_meta_path(), 'w') as outfile:
            json.dump(meta, outfile)

    def read_meta(self):
        with open(self.get_meta_path()) as json_file:
            meta = json.load(json_file)
        return meta

    def len(self):
        if self.previously_processed():
            meta = self.read_meta()
            return meta["length"]  # real length
        else:
            """
            upper bound
            real length may be lower if some data points fail conversion
            """
            for count, _ in enumerate(self.batch_reader.get_batches()):
                pass
            return count

    def get_data_file_path(self, idx):
        return os.path.join(self.processed_dir, 'data_{}.pt'.format(idx))

    def get(self, idx):
        data = torch.load(self.get_data_file_path(idx))
        return data


if __name__ == "__main__":
    cgc = CompoundGraphConstructor([EntitiesConstructor, SequentialEntityLinker, CoreferenceConstructor,
                                    SentenceConstructor, PassageConstructor, DocumentNodeConstructor])
    # cgc = CompoundGraphConstructor([EntitiesConstructor, CoreferenceConstructor, SentenceConstructor,
    #                                 DocumentNodeConstructor])
    from Datasets.Readers.squad_reader import SQuADDatasetReader
    from Datasets.Readers.qangaroo_reader import QUangarooDatasetReader

    squad_reader = SQuADDatasetReader("SQuAD")
    qangaroo_reader = QUangarooDatasetReader("wikihop")

    wikihop_path = QUangarooDatasetReader.dev_set_location("wikihop")
    squad_path = SQuADDatasetReader.dev_set_location()

    qangaroo_batch_reader = BatchReader(qangaroo_reader, 1, wikihop_path)
    squad_batch_reader = BatchReader(squad_reader, 1, squad_path)

    graph_dataset = GraphDataset(squad_batch_reader, cgc)
    # graph_dataset = GraphDataset(qangaroo_batch_reader, cgc)

    loader = DataLoader(graph_dataset, batch_size=256, shuffle=True)
    model = PropAndPool(1536).to(device)

    for i, batch in enumerate(loader):
        print(batch)
        batch=batch.to(device)
        out = model(batch)
