from typing import Tuple

from Code.constants import ENTITY, SENTENCE, TOKEN, PASSAGE, QUERY, CANDIDATE, DOCUMENT


class HDENode:

    # todo implement is_spaecial_entity. Maybe allow for both special and non special entities

    def __init__(self, type, doc_id=None, candidate_id=None, text=None,
                 ent_token_spen: Tuple[int] = None, ent_id=None, is_special_ent=False):
        assert type in [ENTITY, SENTENCE, TOKEN, PASSAGE, CANDIDATE, QUERY, DOCUMENT], "unrecognised type: " + repr(type)
        assert (ent_token_spen is None and type != ENTITY) or \
               (ent_token_spen is not None and type == ENTITY)  # is or isn't an entity
        assert (type == CANDIDATE) == (candidate_id is not None)  # is or isnt a candidate

        self.is_special_ent = is_special_ent  # if this ent is a query/candidate ent via exact match.
        self.ent_id = ent_id  # which in-order entity this node represents. Only relevant if is an ent
        self.token_spen = ent_token_spen  # token span wrt a passage. Only relevant for ents/sentences
        self.text = text
        self.doc_id = doc_id
        self.candidate_id = candidate_id
        self.type = type
        self.id_in_graph = None