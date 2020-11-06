from transformers import TokenSpan

from Code.Data.Graph.Nodes.word_node import WordNode


class CorefNode(WordNode):

    def __init__(self, span: TokenSpan, referenced_node: WordNode):
        super().__init__(span)
        self.referenced_node = referenced_node  # which node this coref is coming from