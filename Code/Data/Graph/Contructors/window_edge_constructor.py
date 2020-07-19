from typing import Union

from Code.Config import config, configuration
from Code.Data.Graph.Contructors.document_structure_constructor import DocumentStructureConstructor
from Code.Data.Graph.Edges.window_edge import WindowEdge
from Code.Data.Graph.context_graph import ContextGraph
from Code.Data.Text.Tokenisation import TokenSpanHierarchy
from Code.Data.Text.data_sample import DataSample


class WindowEdgeConstructor(DocumentStructureConstructor):

    def append(self, existing_graph: Union[None, ContextGraph], data_sample: DataSample,
               context_span_hierarchy: TokenSpanHierarchy) -> ContextGraph:

        level_indices = self.level_indices
        for level_id in level_indices:
            self.connect_level(existing_graph, context_span_hierarchy, level_id)

        existing_graph.constructs.append(type(self))
        return existing_graph

    def connect_level(self, existing_graph, context_span_hierarchy, level_id):
        spans = context_span_hierarchy[level_id]
        level_name = configuration.LEVELS[level_id]
        connection_config = config.structure_connections[level_name]
        
        window_size = connection_config[configuration.WINDOW_SIZE]
        if connection_config[configuration.CONNECTION_TYPE] == configuration.SEQUENTIAL:
            self.connect_level_sequentially(existing_graph, spans, window_size, level_id)
        if connection_config[configuration.CONNECTION_TYPE] == configuration.WINDOW:
            self.connect_level_in_window(existing_graph, spans, window_size, level_id)

    def connect_level_sequentially(self, existing_graph: ContextGraph, spans, window_size, level_id):
        self._connect(existing_graph, spans, window_size, 1, level_id)

    def connect_level_in_window(self, existing_graph: ContextGraph, spans, window_size, level_id):
        self._connect(existing_graph, spans, window_size, -1, level_id)

    @staticmethod
    def _connect(existing_graph: ContextGraph, spans, window_size, max_connections, level_id):
        edges = []
        level_name = configuration.LEVELS[level_id]
        type = config.structure_connections[level_name][configuration.CONNECTION_TYPE]
        for i in range(len(spans) - 1):
            span1 = spans[i]

            for j in range(i + 1, len(spans)):
                if j-i > max_connections != -1:
                    # enough connections made
                    break

                span2 = spans[j]
                if span1.distance(span2) > window_size != -1:
                    # window distnace in token span exceeded
                    break
                try:
                    ids = existing_graph.span_nodes[span1], existing_graph.span_nodes[span2]
                except Exception as e:
                    print("failed to find span nodes in existing graph",existing_graph,existing_graph.span_nodes)
                    raise e
                edges.append(WindowEdge(ids[0], ids[1], type, level_id))

        existing_graph.add_edges(edges)
