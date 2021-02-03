from typing import Tuple


class HDENode:

    def __init__(self, type, doc_id=None, candidate_id=None, text=None, ent_token_spen: Tuple[int] = None, ent_id=None):
        self.ent_id = ent_id  # which in-order entity this node represents
        self.ent_token_spen = ent_token_spen
        self.text = text
        self.doc_id = doc_id
        self.candidate_id = candidate_id
        self.type = type
        self.id_in_graph = None