import Code.constants
from Code.Data.Graph.Contructors.graph_constructor import GraphConstructor
from Code.Data.Graph.Edges.candidate_edge import CandidateEdge
from Code.Data.Graph.Nodes.candidate_node import CandidateNode
from Code.Data.Graph.context_graph import ContextGraph


class CandidatesConstructor(GraphConstructor):
    def _append(self, existing_graph: ContextGraph) -> ContextGraph:
        # todo multiple questions
        candidates = existing_graph.data_sample.questions[0].answers.answer_candidates
        if candidates is None:
            # print("no candidates, cannot add candidate nodes")
            return existing_graph
        candidate_sequences = [cand.token_sequence for cand in candidates]
        candidate_spans = [TokenSpan(seq, (0, len(seq))) for seq in candidate_sequences]
        candidate_nodes = [CandidateNode(candidate_spans[s], s) for s in range(len(candidate_spans))]

        node_ids = existing_graph.add_nodes(candidate_nodes)

        # need to connect the candidate nodes to both the context and query nodes
        self.connect_to_context(existing_graph, node_ids)
        self.connect_to_query(existing_graph, node_ids)

        self.add_construct(existing_graph)
        return existing_graph

    def connect_to_context(self, existing_graph, node_ids):
        connection_levels = existing_graph.gcc.candidate_connections
        if connection_levels == [Code.constants.GLOBAL]:
            # connect to all context nodes
            connection_levels = existing_graph.gcc.context_structure_levels
        for connection_level in connection_levels:
            if connection_level not in existing_graph.gcc.context_structure_levels:
                raise Exception("cannot connect candidate nodes to context at " + connection_level +
                                " level as this level is not being graphed in the context. Only gaphing: " +
                                repr(existing_graph.gcc.context_structure_levels))

            # todo code reuse with query constructor. make generic extra-context constructor with context conn method
            context_ids = existing_graph.get_context_node_ids_at_level(connection_level)
            for node_id in node_ids:
                edges = [CandidateEdge(node_id, con_id, connection_level, Code.constants.CONTEXT) for con_id in context_ids]
                existing_graph.add_edges(edges)

    def connect_to_query(self, existing_graph, node_ids):
        connection_levels = existing_graph.gcc.query_structure_levels
        for connection_level in connection_levels:
            # todo code reuse with query constructor. make generic extra-context constructor with context conn method
            query_ids = existing_graph.get_query_node_ids_at_level(connection_level)
            for node_id in node_ids:
                edges = [CandidateEdge(node_id, q_id, connection_level, Code.constants.QUERY) for q_id in query_ids]
                existing_graph.add_edges(edges)

