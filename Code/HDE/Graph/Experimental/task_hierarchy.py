from typing import List

from Code.HDE.Graph.Experimental.context_hierarchy import ContextHierarchy
from Code.HDE.Graph.edge import HDEEdge
from Code.HDE.Graph.graph import HDEGraph
from Code.HDE.Graph.node import HDENode
from Code.Utils.graph_utils import clean, fully_connect, connect_one_to_all
from Code.constants import CANDIDATE, PASSAGE, FULLY_CONNECT, QUERY, SUBQUERY


class TaskHierarchy:

    """
        represents a full problem in node form.

        Contains  a context hierarchy, as well as optional candidate, query/subquery nodes
        A subquery is a node generated via coattention between the query and a particular passage.
        IE a subquery is a passage-aware query rep. There are thus multiple subqueries per query. 1 for each passage
    """

    def __init__(self):
        self.context: ContextHierarchy = None

    def get_all_node_ids(self, graph: HDEGraph):
        ctx_ids = self.context.get_all_node_ids()
        candidate_ids = [n.id_in_graph for n in graph.get_candidate_nodes()]
        ctx_ids.extend(candidate_ids)
        # todo get query node ids
        return ctx_ids

    def add_candidate_nodes(self, graph: HDEGraph, candidates: List[str], supports: List[str]):
        """
            candidate nodes are fully connected with eachother.
            there is also a special edge between a candidate node,  and any passage it is mentioned in
        """
        cand_ids = []
        cand2pass = CANDIDATE + "_in_" + PASSAGE
        fc_cands = FULLY_CONNECT + "_" + CANDIDATE

        for c, candidate in enumerate(candidates):
            node = HDENode(CANDIDATE, candidate_id=c, text=candidate)
            c_id = graph.add_node(node)

            cand_ids.append(c_id)

            cand_node = graph.get_candidate_nodes()[c]
            clean_cand = clean(candidate)
            for s, supp in enumerate(supports):
                supp = clean(supp)
                if clean_cand in supp:
                    edge = HDEEdge(s, cand_node.id_in_graph, graph=graph, type=cand2pass)  # type 1
                    graph.add_edge(edge)

        fully_connect(cand_ids, graph, fc_cands)  # type 6

    def add_query_nodes(self, graph: HDEGraph, query: str, supports: List[str]):
        """
            this should be called only after all other nodes have been added
            the query node is connected to all other nodes.
            subquery nodes are connnected to all context nodes for their corresponding passage
            they are also connected to all candidate nodes
        """
        for s, supp in enumerate(supports):  # for each passage, create a subquery node
            sq_node = HDENode(SUBQUERY, doc_id=s)
            sq_id = graph.add_node(sq_node)
            passage_nodes = self.context.passages[s].get_all_node_ids()
            connect_one_to_all(sq_id, passage_nodes, graph)  # connect subquery to all of its context nodes
            candidate_ids = [n.id_in_graph for n in graph.get_candidate_nodes()]
            connect_one_to_all(sq_id, candidate_ids, graph)  # connect subquery to all candidate nodes

        q_node = HDENode(QUERY, text=query)
        q_id = graph.add_node(q_node)
        all_ids = self.get_all_node_ids(graph)
        connect_one_to_all(q_id, all_ids, graph)
