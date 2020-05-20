from typing import Union

from Code.Data.Graph.Contructors.coreference_constructor import CoreferenceConstructor
from Code.Data.Graph.Contructors.entities_constructor import EntitiesConstructor
from Code.Data.Graph.Contructors.graph_constructor import GraphConstructor, IncompatibleGraphContructionOrder
from Code.Data.Graph.Contructors.sentence_contructor import SentenceConstructor
from Code.Data.Graph.context_graph import ContextGraph
from Code.Data.Text.data_sample import DataSample


class PassageConstructor(GraphConstructor):

    """
    creates nodes which represent entire passage summaries and links them to the entities contained.
    if the given graph contains sentence nodes, these are linked to instead of the entities
    """

    def append(self, existing_graph: Union[None, ContextGraph], data_sample: DataSample) -> ContextGraph:
        if not existing_graph or EntitiesConstructor not in existing_graph.constructs \
                or CoreferenceConstructor not in existing_graph.constructs:
            raise IncompatibleGraphContructionOrder(existing_graph, self,
                                                    "Entities and corefs must be graphed before sentences")

        if SentenceConstructor in existing_graph.constructs:
            return link_to_sentences()
