from typing import Type, List

from Code.Config import graph_construction_config as construction
from Code.Data.Graph.Contructors.graph_constructor import GraphConstructor
from Code.Data.Graph.Edges.document_edge import DocumentEdge
from Code.Data.Graph.Nodes.document_structure_node import DocumentStructureNode
from Code.Data.Graph.Nodes.entity_node import EntityNode
from Code.Data.Graph.Nodes.span_node import SpanNode
from Code.Data.Graph.Nodes.token_node import TokenNode
from Code.Data.Graph.context_graph import ContextGraph
from Code.Data.Text.Tokenisation.document_extract import DocumentExtract
from Code.Data.Text.Tokenisation.entity_span import EntitySpan
from Code.Data.Text.Tokenisation.token import Token


def get_node_type(span) -> Type[SpanNode]:
    if isinstance(span, EntitySpan):
        return EntityNode
    if isinstance(span, Token):
        return TokenNode
    if isinstance(span, DocumentExtract):
        return DocumentStructureNode

    raise Exception("cannot find type of span: " + repr(type(span)) + "-" +repr(span))


class DocumentStructureConstructor(GraphConstructor):

    def level_indices(self, existing_graph: ContextGraph):
        level_indices = [construction.LEVELS.index(level) for level in existing_graph.gcc.context_structure_nodes]
        level_indices = sorted(level_indices)
        return level_indices

    def _append(self, existing_graph: ContextGraph) -> ContextGraph:
        level_indices = self.level_indices(existing_graph)
        for i in range(len(level_indices) - 1):
            """
            loops through each container span, creates a node for it and each of its contained spans, 
            then connects the nodes via a DocumentEdge
            """
            containeD_spans : List[DocumentExtract] = existing_graph.span_hierarchy[level_indices[i]]
            containeR_spans : List[DocumentExtract] = existing_graph.span_hierarchy[level_indices[i+1]]

            try:
                contain_map = existing_graph.span_hierarchy.match_heirarchical_span_seqs(containeR_spans, containeD_spans)
            except Exception as e:
                print("failed matching", construction.LEVELS[level_indices[i]], "to",
                      construction.LEVELS[level_indices[i + 1]])
                raise e

            # print("contain map:", contain_map)
            for containeR in containeR_spans:  # container will never be a token sequence
                node_type = get_node_type(containeR)
                containeR_node = node_type(containeR, subtype=containeR.get_subtype())
                Rid = existing_graph.add_node(containeR_node)
                # print("container type:",containeR.get_subtype(), "-", containeR)

                for containeD in contain_map[containeR]:  # for each contained span
                    node_type = get_node_type(containeD)
                    containeD_node = node_type(containeD, subtype=containeD.get_subtype())
                    Did = existing_graph.add_node(containeD_node)
                    # print("contained type:", containeD.get_subtype(),"-",containeD)

                    edge_subtype = DocumentEdge.get_x2y_edge_type(containeR.level, containeD.level)
                    existing_graph.add_edge(DocumentEdge(Rid, Did, edge_subtype))

        self.add_construct(existing_graph)
        return existing_graph


if __name__ == "__main__":
    from Datasets.Readers.squad_reader import SQuADDatasetReader
    from Datasets.Readers.qangaroo_reader import QUangarooDatasetReader

    sq_reader = SQuADDatasetReader("SQuAD")
    qangaroo_reader = QUangarooDatasetReader("wikihop")

    reader = qangaroo_reader
    # reader = sq_reader

    samples = reader.get_dev_set()

    const = DocumentStructureConstructor()

    for i, sample in enumerate(samples):
        if i >= 5:
            break

        graph = const._append(None, sample)
        print("num nodes:", len(graph.ordered_nodes))
        graph.render_graph(sample.title_and_peek, reader.datset_name)