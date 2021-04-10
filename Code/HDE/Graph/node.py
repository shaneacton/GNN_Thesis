from typing import Tuple


class HDENode:

    def __init__(self, type, doc_id=None, candidate_id=None, text=None,
                 ent_token_spen: Tuple[int] = None, ent_id=None, is_special_ent=False):
        self.is_special_ent = is_special_ent  # if this ent is a query/candidate ent via exact match.
        self.ent_id = ent_id  # which in-order entity this node represents
        self.ent_token_spen = ent_token_spen
        self.text = text
        self.doc_id = doc_id
        self.candidate_id = candidate_id
        self.type = type
        self.id_in_graph = None
