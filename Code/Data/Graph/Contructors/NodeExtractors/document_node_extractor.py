from typing import List

from Code.Data.Graph.Contructors.NodeExtractors.node_extractor import NodeExtractor
from Code.Data.Graph.Nodes.node import Node
from Code.Data.Text.context import Context


class DocumentNodeExtractor(NodeExtractor):

    """
    document node extractor creates a node for every token in the given context

    it also optionally adds in Statement-Level-Nodes, Passage-Level-Nodes, and a single
    Doc-Level_Node
    """

    def __init__(self, context: Context):
        super().__init__(context)

    def extract_nodes(self) -> List[Node]:
        pass

