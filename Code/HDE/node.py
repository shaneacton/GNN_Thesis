
class HDENode:

    DOC = "doc"
    ENTITY = "entity"
    CANDIDATE = "candidate"

    def __init__(self, type, doc_id=None, candidate_id=None):
        self.doc_id = doc_id
        self.candidate_id = candidate_id
        self.type = type
