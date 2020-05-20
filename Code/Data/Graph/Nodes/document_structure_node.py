from Code.Data.Graph.Nodes.entity_node import EntityNode
from Code.Data.Text.Tokenisation.document_extract import DocumentExtract


class DocumentStructureNode(EntityNode):

    """can represent a sentence, passage, or whole document"""

    def __init__(self, sentence: DocumentExtract):
        super().__init__(sentence)