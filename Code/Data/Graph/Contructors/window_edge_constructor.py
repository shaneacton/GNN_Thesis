from Code.Config import graph_construction_config as construction
from Code.Data.Graph.Contructors.document_structure_constructor import DocumentStructureConstructor
from Code.Data.Graph.Edges.window_edge import WindowEdge
from Code.Data.Graph.context_graph import ContextGraph
from Code.Data.Text.Tokenisation.token_span_hierarchy import TokenSpanHierarchy


class WindowEdgeConstructor(DocumentStructureConstructor):

    def _append(self, existing_graph: ContextGraph) -> ContextGraph:

        level_indices = self.level_indices(existing_graph)
        for level_id in level_indices:
            self.connect_level(existing_graph, existing_graph.span_hierarchy, level_id)

        self.add_construct(existing_graph)
        return existing_graph

    def connect_level(self, existing_graph, context_span_hierarchy: TokenSpanHierarchy, level_id):
        spans = context_span_hierarchy[level_id]
        level_name = construction.LEVELS[level_id]
        connection_config = existing_graph.gcc.structure_connections[level_name]
        
        window_size = connection_config[construction.WINDOW_SIZE]
        if connection_config[construction.CONNECTION_TYPE] == construction.SEQUENTIAL:
            self.connect_level_sequentially(existing_graph, spans, window_size, level_id)
        if connection_config[construction.CONNECTION_TYPE] == construction.WINDOW:
            self.connect_level_in_window(existing_graph, spans, window_size, level_id)

    def connect_level_sequentially(self, existing_graph: ContextGraph, spans, window_size, level_id):
        self._connect(existing_graph, spans, window_size, 1, level_id)

    def connect_level_in_window(self, existing_graph: ContextGraph, spans, window_size, level_id):
        self._connect(existing_graph, spans, window_size, -1, level_id)

    @staticmethod
    def _connect(existing_graph: ContextGraph, spans, window_size, max_connections, level_id):
        """
        generic window connection method.
        connects nodes on the same level with each other in an optional window size
        when max connections != -1, only the first m_c subsequent nodes will be connected
        setting max connections to 1 makes the connections sequential only
        """
        edges = []
        level_name = construction.LEVELS[level_id]
        type = existing_graph.gcc.structure_connections[level_name][construction.CONNECTION_TYPE]
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