from Code.Config import gcc, graph_construction_config
from Code.Data.Graph.Contructors.graph_constructor import GraphConstructor
from Code.Data.Graph.Edges.query_edge import QueryEdge
from Code.Data.Graph.Nodes.query_node import QueryNode
from Code.Data.Graph.context_graph import ContextGraph
from Code.Data.Text.Tokenisation.token_span import TokenSpan


class QueryConstructor(GraphConstructor):
    def _append(self, existing_graph: ContextGraph) -> ContextGraph:
        if graph_construction_config.QUERY_SENTENCE in gcc.query_node_types:
            self.add_sentence_query_node(existing_graph)
        if graph_construction_config.QUERY_TOKENS in gcc.query_node_types:
            self.add_token_query_nodes(existing_graph)
        if graph_construction_config.QUERY_ENTITIES in gcc.query_node_types:
            self.add_entity_query_nodes(existing_graph)

        self.add_construct(existing_graph)
        return existing_graph

    def add_sentence_query_node(self, existing_graph):
        query_seq = existing_graph.query_token_sequence
        query_span = TokenSpan(query_seq, (0, len(query_seq)))
        self.create_and_connect_query_node(existing_graph, query_span, graph_construction_config.QUERY_SENTENCE)

    def add_token_query_nodes(self, existing_graph):
        query_seq = existing_graph.query_token_sequence
        for query_token in query_seq.subtokens:
            self.create_and_connect_query_node(existing_graph, query_token, graph_construction_config.QUERY_TOKENS)

    def add_entity_query_nodes(self, existing_graph):
        raise NotImplementedError()

    def create_and_connect_query_node(self, existing_graph, query_span, query_level):
        query_node = QueryNode(query_span, query_level)
        query_id = existing_graph.add_node(query_node)
        connection_levels = gcc.query_connections[query_level]

        if connection_levels == [graph_construction_config.GLOBAL]:
            # connect to all context nodes
            connection_levels = gcc.structure_nodes

        for connection_level in connection_levels:
            if connection_level not in gcc.structure_nodes:
                raise Exception("cannot connect " + query_level + " query node to " + connection_level +
                                "context nodes as these nodes don't exist. Try adding in "
                                + connection_level + " graph structuring")

            context_ids = existing_graph.get_context_node_ids_at_level(connection_level)
            edges = [QueryEdge(query_id, con_id, query_level, connection_level) for con_id in context_ids]
            existing_graph.add_edges(edges)
