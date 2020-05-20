from typing import Union

from Code.Data.Graph.Contructors.coreference_constructor import CoreferenceConstructor
from Code.Data.Graph.Contructors.entities_constructor import EntitiesConstructor
from Code.Data.Graph.Contructors.graph_constructor import GraphConstructor, IncompatibleGraphContructionOrder
from Code.Data.Graph.Edges.document_edge import DocumentEdge
from Code.Data.Graph.Nodes.entity_node import EntityNode
from Code.Data.Graph.Nodes.document_structure_node import DocumentStructureNode
from Code.Data.Graph.context_graph import ContextGraph
from Code.Data.Text.data_sample import DataSample


class SentenceConstructor(GraphConstructor):

    """
    creates nodes which represent entire sentence summaries and links these nodes to the contained entity mentions and corefs
    """

    def append(self, existing_graph: Union[None, ContextGraph], data_sample: DataSample) -> ContextGraph:
        if not existing_graph or EntitiesConstructor not in existing_graph.constructs \
                or CoreferenceConstructor not in existing_graph.constructs:
            raise IncompatibleGraphContructionOrder(existing_graph, self,
                                                    "Entities and corefs must be graphed before sentences")

        for sentence in data_sample.context.token_sequence.sentences:
            node = DocumentStructureNode(sentence)
            sentence_node_id = existing_graph.add_node(node)

            contained_entity_nodes = [EntityNode(ent) for ent in sentence.contained_entities]
            contained_entity_node_ids = existing_graph.add_nodes(contained_entity_nodes)

            edges = [DocumentEdge(sentence_node_id, cont_node_id, subtype="stat2word")
                     for cont_node_id in contained_entity_node_ids]

            existing_graph.add_edges(edges)

        existing_graph.constructs.append(type(self))
        return existing_graph




