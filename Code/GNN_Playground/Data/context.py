from typing import List, Union

from Code.GNN_Playground.Data.passage import Passage


class Context:
    """
        a collection of passages with a natural grouping
        ie a document
    """

    def __init__(self,passages : Union[List[Passage], Passage, None] = None):
        if isinstance(passages, List):
            self.passages : List[Passage] = passages
        if isinstance(passages, Passage):
            self.passages : List[Passage] = [passages]
        if passages is None:
            self.passages : List[Passage] = []

    def add_passage(self, passage: Passage):
        self.passages.append(passage)

    def add_text_as_passage(self, text):
        self.passages.append(Passage(text))

    def get_all_text_pieces(self):
        return [passage.text for passage in self.passages]

    def get_full_context(self):
        return "\n\n".join(self.get_all_text_pieces())