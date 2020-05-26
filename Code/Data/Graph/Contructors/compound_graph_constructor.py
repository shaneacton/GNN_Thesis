from typing import Union, List

from Code.Data.Graph.Contructors.coreference_constructor import CoreferenceConstructor
from Code.Data.Graph.Contructors.document_node_constructor import DocumentNodeConstructor
from Code.Data.Graph.Contructors.entities_constructor import EntitiesConstructor
from Code.Data.Graph.Contructors.graph_constructor import GraphConstructor
from Code.Data.Graph.Contructors.passage_constructor import PassageConstructor
from Code.Data.Graph.Contructors.sentence_contructor import SentenceConstructor
from Code.Data.Graph.Contructors.sequential_entity_linker import SequentialEntityLinker
from Code.Data.Graph.context_graph import ContextGraph
from Code.Data.Text.data_sample import DataSample


class CompoundGraphConstructor(GraphConstructor):

    def __init__(self, constructors: List[type]):
        self.constructors = constructors

    @property
    def type(self):
        return ",".join([repr(const) for const in self.constructors])

    def append(self, _existing_graph: Union[None, ContextGraph], data_sample: DataSample) -> ContextGraph:
        existing_graph = ContextGraph()
        for const_type in self.constructors:
            # print("constructing using:", const_type)
            constructor: GraphConstructor = const_type()
            existing_graph = constructor.append(existing_graph, data_sample)
            if not existing_graph:
                raise Exception()

        return existing_graph


if __name__ == "__main__":
    cgc = CompoundGraphConstructor([EntitiesConstructor, SequentialEntityLinker, CoreferenceConstructor,
                                    SentenceConstructor, PassageConstructor, DocumentNodeConstructor])
    # cgc = CompoundGraphConstructor([EntitiesConstructor])
    from Datasets.Readers.squad_reader import SQuADDatasetReader
    from Datasets.Readers.qangaroo_reader import QUangarooDatasetReader

    sq_reader = SQuADDatasetReader("SQuAD")
    qangaroo_reader = QUangarooDatasetReader("wikihop")

    reader = qangaroo_reader
    # reader = sq_reader

    samples = reader.get_dev_set()

    for i, sample in enumerate(samples):
        if i>=5:
            break

        graph = cgc.append(None, sample)
        print(sample)
        print("num nodes:", len(graph.ordered_nodes))
        graph.render_graph(sample.title_and_peek, reader.datset_name)


