from typing import List

from Code.Config import config
from Code.Data.Graph.Contructors.graph_constructor import GraphConstructor
from Code.Data.Graph.context_graph import ContextGraph


class CompoundGraphConstructor(GraphConstructor):

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
    cgc = config.get_graph_constructor()
    from Datasets.Readers.squad_reader import SQuADDatasetReader
    from Datasets.Readers.qangaroo_reader import QUangarooDatasetReader

    sq_reader = SQuADDatasetReader("SQuAD")
    qangaroo_reader = QUangarooDatasetReader("wikihop")

    # reader = qangaroo_reader
    reader = sq_reader

    samples = reader.get_dev_set()

    for i, sample in enumerate(samples):
        if i>=3:
            break

        graph = cgc.create_graph_from_data_sample(data_sample=sample)
        # print("sample:", sample)
        # print("num nodes:", len(graph.ordered_nodes))
        graph.render_graph(sample.title_and_peek, reader.datset_name)


