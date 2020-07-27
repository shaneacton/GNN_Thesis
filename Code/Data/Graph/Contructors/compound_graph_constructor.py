from typing import List

import torch

from Code.Data.Graph.Contructors.graph_constructor import GraphConstructor
from Code.Data.Graph.context_graph import ContextGraph
from Code.Models.GNNs.context_gnn import ContextGNN
from Code.Training import device


class CompoundGraphConstructor(GraphConstructor):

    """
    passes a graph through multiple constructors in order
    """

    def __init__(self, constructors: List[type]):
        self.constructors = constructors

    @property
    def type(self):
        return ",".join([repr(const) for const in self.constructors])

    def _append(self, existing_graph: ContextGraph) -> ContextGraph:
        for const_type in self.constructors:
            constructor: GraphConstructor = const_type()
            existing_graph = constructor._append(existing_graph)
            if not existing_graph:
                raise Exception()

        return existing_graph


if __name__ == "__main__":
    from Datasets.Readers.squad_reader import SQuADDatasetReader
    from Datasets.Readers.qangaroo_reader import QUangarooDatasetReader

    from Code.Config import gcc, gec, gnnc

    sq_reader = SQuADDatasetReader("SQuAD")
    qangaroo_reader = QUangarooDatasetReader("wikihop")

    # reader = qangaroo_reader
    reader = sq_reader
    samples = reader.get_dev_set()

    cgc = gcc.get_graph_constructor()
    embedder = gec.get_graph_embedder(gcc).to(device)
    gnn = ContextGNN(embedder, gnnc).to(device=device)

    for i, sample in enumerate(samples):
        if i>=3:
            break

        graph = cgc.create_graph_from_data_sample(data_sample=sample)

        # embedding = embedder(graph)
        #
        # print("embedder:", embedder,"dev:", next(embedder.parameters()).device)
        # print("embedding:", embedding, "dev:", embedding.x.device)

        out = gnn(graph)
        print("gnn", gnn, "out:", out)


        # print("sample:", sample)
        # print("num nodes:", len(graph.ordered_nodes))
        graph.render_graph(sample.title_and_peek, reader.datset_name)


