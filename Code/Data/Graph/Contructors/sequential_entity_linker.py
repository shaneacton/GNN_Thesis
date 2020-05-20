from typing import Union

from Code.Data.Graph.Contructors.entities_constructor import EntitiesConstructor
from Code.Data.Graph.Contructors.graph_constructor import GraphConstructor, IncompatibleGraphContructionOrder
from Code.Data.Graph.Edges.adjacent_entity_edge import AdjacentEntityEdge
from Code.Data.Graph.context_graph import ContextGraph
from Code.Data.Text.data_sample import DataSample


class SequentialEntityLinker(GraphConstructor):

    """
    connects sequential entity mentions within a maximum sized window
    """

    MAX_DISTANCE = 20

    def append(self, existing_graph: Union[None, ContextGraph], data_sample: DataSample) -> ContextGraph:
        if not existing_graph or EntitiesConstructor not in existing_graph.constructs:
            raise IncompatibleGraphContructionOrder(existing_graph, self,
                                                    "Entities must be graphed before sequentially linked")

        #todo possibly insert coref ents in for connection
        entities = data_sample.context.token_sequence.entities

        edges = []
        for i in range(len(entities) -1):
            ent1 = entities[i]
            ent2 = entities[i+1]
            if ent1.distance(ent2) > SequentialEntityLinker.MAX_DISTANCE:
                continue
            ids = existing_graph.span_nodes[ent1], existing_graph.span_nodes[ent2]
            edges.append(AdjacentEntityEdge(ids[0], ids[1]))

        existing_graph.add_edges(edges)
        existing_graph.constructs.append(type(self))
        return existing_graph
