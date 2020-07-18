from typing import Union, Type, List

from Code.Config import config
from Code.Data.Graph.Contructors.graph_constructor import GraphConstructor
from Code.Data.Graph.Edges.document_edge import DocumentEdge
from Code.Data.Graph.Nodes.document_structure_node import DocumentStructureNode
from Code.Data.Graph.Nodes.entity_node import EntityNode
from Code.Data.Graph.Nodes.span_node import SpanNode
from Code.Data.Graph.context_graph import ContextGraph
from Code.Data.Text.Tokenisation import TokenSpanHierarchy
from Code.Data.Text.Tokenisation.document_extract import DocumentExtract
from Code.Data.Text.Tokenisation.entity_span import EntitySpan
from Code.Data.Text.data_sample import DataSample


def get_node_type(span) -> Type[SpanNode]:
    if isinstance(span, EntitySpan):
        return EntityNode
    if isinstance(span, DocumentExtract):
        return DocumentStructureNode
    raise Exception("cannot find type of span: " + repr(type(span)) + "-" +repr(span))


class NewDocumentStructureConstructor(GraphConstructor):

    def append(self, existing_graph: Union[None, ContextGraph], data_sample: DataSample) -> ContextGraph:
        if not existing_graph:
            existing_graph = ContextGraph()

        tok_seq = data_sample.context.token_sequence
        span_hierarchy = TokenSpanHierarchy(tok_seq)

        level_indices = [DocumentExtract.levels.index(level) for level in config.document_structure_nodes]
        level_indices = sorted(level_indices)
        print("based on config", config.document_structure_nodes, "level indices=",level_indices)

        for i in range(len(level_indices) - 1):
            """
            loops through each container span, creates a node for it and each of its contained spans, 
            then connects the nodes via a DocumentEdge
            """
            containeD_spans = span_hierarchy[level_indices[i]]
            containeR_spans : List[DocumentExtract] = span_hierarchy[level_indices[i+1]]

            contain_map = span_hierarchy.match_heirarchical_span_seqs(containeR_spans, containeD_spans)

            for containeR in containeR_spans:  # container will never be a token sequence
                node_type = get_node_type(containeR)
                containeR_node = node_type(containeR, subtype=containeR.get_subtype())
                Rid = existing_graph.add_node(containeR_node)
                print("container type:",containeR.get_subtype(), "-", containeR)

                for containeD in contain_map[containeR]:  # for each contained span
                    node_type = get_node_type(containeD)
                    containeD_node = node_type(containeD, subtype=containeD.get_subtype())
                    Did = existing_graph.add_node(containeD_node)
                    print("contained type:", containeD.get_subtype(),"-",containeD)

                    edge_subtype = DocumentEdge.get_x2y_edge_type(containeR.level, containeD.level)
                    existing_graph.add_edge(DocumentEdge(Rid, Did, edge_subtype))

        return existing_graph

if __name__ == "__main__":
    from Datasets.Readers.squad_reader import SQuADDatasetReader
    from Datasets.Readers.qangaroo_reader import QUangarooDatasetReader

    sq_reader = SQuADDatasetReader("SQuAD")
    qangaroo_reader = QUangarooDatasetReader("wikihop")

    reader = qangaroo_reader
    # reader = sq_reader

    samples = reader.get_dev_set()

    const = NewDocumentStructureConstructor()

    for i, sample in enumerate(samples):
        if i >= 5:
            break

        graph = const.append(None, sample)
        print("num nodes:", len(graph.ordered_nodes))
        graph.render_graph(sample.title_and_peek, reader.datset_name)