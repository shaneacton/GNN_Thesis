import json
import os
from abc import ABC, abstractmethod
from typing import Iterator

from torch_geometric.data import Dataset

from Code.Data.Graph.Contructors.compound_graph_constructor import CompoundGraphConstructor
from Code.Data.Graph.context_graph import ContextGraph
from Datasets.Batching.batch_reader import BatchReader


def get_geometric_dataset_folder():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "..", "..", "Datasets", "Geometric")


class GraphDataset(Dataset, ABC):

    """yields context graphs either live or reprocessed"""

    ROOT = get_geometric_dataset_folder()

    def __init__(self, batch_reader: BatchReader, graph_constructor: CompoundGraphConstructor):
        self.graph_constructor = graph_constructor
        self.batch_text_reader = batch_reader

        if batch_reader.batch_size > 1:
            raise Exception("must provide batch reader with size 1 to graph dataset")

        super(GraphDataset, self).__init__(self.variant_path, None, None)
        # process is called by super

    @property
    def dataset_path(self):
        return os.path.join(GraphDataset.ROOT, self.batch_text_reader.dataset)

    @property
    def variant_path(self):
        return os.path.join(self.dataset_path, self.variant_id)

    def get_data_file_path(self, idx):
        return os.path.join(self.processed_dir, 'data_{}.pt'.format(idx))

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data_{}.pt'.format(i) for i in range(len(self))]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    @abstractmethod
    def get_context_graphs(self) -> Iterator[ContextGraph]:
        """loads/constructs the context graphs, and then encodes and yields the data points"""

    def get_meta_path(self):
        return os.path.join(self.processed_dir, "meta.json")

    def write_meta(self, length):
        meta = {"length": length,"dataset":self.batch_text_reader.dataset, "graphtype": self.graph_constructor.type}
        with open(self.get_meta_path(), 'w') as outfile:
            json.dump(meta, outfile)

    def read_meta(self):
        with open(self.get_meta_path()) as json_file:
            meta = json.load(json_file)
        return meta

    def write_variant_map(self, map):
        with open(self.get_variant_map_path(), 'w') as outfile:
            json.dump(map, outfile)

    def read_variant_map(self):
        if not os.path.exists(self.get_variant_map_path()):
            return {}
        with open(self.get_variant_map_path()) as json_file:
            map = json.load(json_file)
        return map

    def get_variant_map_path(self):
        return os.path.join(self.dataset_path, "variant_map.json")

    @property
    def variant_id(self):
        variant = self.graph_constructor.type
        map = self.read_variant_map()
        if not variant in map:
            """new variant"""
            next_num = len(list(map.keys()))
            id = "variant_" + str(next_num)
            map[variant] = id
            self.write_variant_map(map)

        return map[variant]

    def len(self):
        if self.previously_processed():
            meta = self.read_meta()
            return meta["length"]  # real length
        else:
            """
            upper bound
            real length may be lower if some data points fail conversion
            """
            for count, _ in enumerate(self.batch_text_reader.get_all_batches()):
                pass
            return count

    def get(self, idx):
        raise Exception()
        # load up the context graph and return the newly created geometric datapoint
        # context_graph =
        # return data

