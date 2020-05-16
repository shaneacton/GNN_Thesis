from typing import Set

from Code.Data.Graph.Contructors.EdgeExtractors.edge_extractor import EdgeExtractor
from Code.Data.Graph.Edges.edge_relation import EdgeRelation


class ComentionExtractor(EdgeExtractor):

    def extract_edges(self) -> Set[EdgeRelation]:
        pass

    def get_edge_type(self, from_node, to_node):
        return