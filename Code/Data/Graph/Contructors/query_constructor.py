from Code.Config import graph_construction_config as construction
from Code.Data.Graph.Contructors.graph_constructor import GraphConstructor
from Code.Data.Graph.Edges.query_edge import QueryEdge
from Code.Data.Graph.Nodes.document_structure_node import DocumentStructureNode
from Code.Data.Graph.Nodes.entity_node import EntityNode
from Code.Data.Graph.Nodes.token_node import TokenNode
from Code.Data.Graph.context_graph import ContextGraph
from Code.Data.Text.Tokenisation.document_extract import DocumentExtract
from Code.Data.Text.Tokenisation.token_span import TokenSpan


class QueryConstructor(GraphConstructor):
    def _append(self, existing_graph: ContextGraph) -> ContextGraph:
        if construction.QUERY_SENTENCE in existing_graph.gcc.query_node_types:
            self.add_sentence_query_node(existing_graph)
        if construction.QUERY_TOKEN in existing_graph.gcc.query_node_types:
            self.add_token_query_nodes(existing_graph)
        if construction.QUERY_WORD in existing_graph.gcc.query_node_types:
            self.add_entity_query_nodes(existing_graph)

        if existing_graph.gcc.fully_connect_query_nodes:
            self.connect_query_nodes(existing_graph)


        self.add_construct(existing_graph)
        return existing_graph

    def add_sentence_query_node(self, existing_graph):
        query_seq = existing_graph.query_token_sequence
        query_span = TokenSpan(query_seq, (0, len(query_seq)))
        self.create_and_connect_query_node(existing_graph, query_span, construction.QUERY_SENTENCE)

    def add_token_query_nodes(self, existing_graph):
        query_seq = existing_graph.query_token_sequence
        for query_token in query_seq.subtokens:
            self.create_and_connect_query_node(existing_graph, query_token, construction.QUERY_TOKEN)

    def add_entity_query_nodes(self, existing_graph):
        ents = existing_graph.query_span_hierarchy.entities_and_corefs \
            if construction.COREF in existing_graph.gcc.word_nodes \
            else existing_graph.query_span_hierarchy.entities

        for ent_span in ents:
            self.create_and_connect_query_node(existing_graph, ent_span, construction.QUERY_WORD)

    def create_and_connect_query_node(self, existing_graph, query_span, query_level):
        """
        :param query_level: {QUERY_TOKEN, QUERY_WORD, QUERY_SENTENCE}
        """
        query_node = self.create_query_node(query_span, query_level)
        query_id = existing_graph.add_node(query_node)
        connection_levels = existing_graph.gcc.query_connections[query_level]

        if connection_levels == [construction.GLOBAL]:
            # connect to all context nodes
            connection_levels = existing_graph.gcc.context_structure_nodes

        for connection_level in connection_levels:
            if connection_level not in existing_graph.gcc.context_structure_nodes:
                raise Exception("cannot connect " + query_level + " query node to " + connection_level +
                                "context nodes as these nodes don't exist. Try adding in "
                                + connection_level + " graph structuring")

            context_ids = existing_graph.get_context_node_ids_at_level(connection_level)
            edges = [QueryEdge(query_id, con_id, query_level, connection_level) for con_id in context_ids]
            existing_graph.add_edges(edges)

    @staticmethod
    def create_query_node(query_span: TokenSpan, query_level):
        if query_level == construction.QUERY_SENTENCE:
            extract = DocumentExtract(query_span.token_sequence, query_span.subtoken_indexes,
                                      level=construction.QUERY_SENTENCE)
            sentence_node = DocumentStructureNode(extract, source=construction.QUERY, subtype=extract.get_subtype())
            return sentence_node

        if query_level == construction.QUERY_TOKEN:
            token_node = TokenNode(query_span, source=construction.QUERY)
            return token_node

        if query_level == construction.QUERY_WORD:
            ent_node = EntityNode(query_span, source=construction.QUERY)
            return ent_node

    def connect_query_nodes(self, existing_graph):
        """fully connects all query nodes together"""
        for node_id_a in existing_graph.query_nodes:
            for node_id_b in existing_graph.query_nodes:
                if node_id_a == node_id_b:
                    continue
                edge = QueryEdge(node_id_a, node_id_b, construction.QUERY, construction.QUERY)
                existing_graph.add_edge(edge)
