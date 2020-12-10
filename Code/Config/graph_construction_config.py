from Code.Config.config import Config
from Code.constants import ENTITY, COREF, TOKEN, WORD, SENTENCE, PARAGRAPH, DOCUMENT, CONNECTION_TYPE, WINDOW, \
    SEQUENTIAL, WINDOW_SIZE, NOUN, CONTEXT, QUERY


class GraphConstructionConfig(Config):

    def __init__(self):

        super().__init__()

        # which structure levels to make nodes for
        # {TOKEN, WORD, SENTENCE, PARAGRAPH, DOCUMENT}
        self.structure_levels = {
            CONTEXT: [NOUN, SENTENCE],
            QUERY: [TOKEN, SENTENCE]
        }

        self.max_edges = 400000  # 400000
        # cuts off all chars after the max. tries to proceed with partial context. -1 to turn off
        self.max_context_chars = -1

    @property
    def all_structure_levels(self):
        levels = self.context_structure_levels + self.query_structure_levels
        if self.use_candidate_nodes:
            levels.append(WORD)
        return set(levels)

    def has_keyword(self, word:str):
        return word in self.word_nodes or word in self.context_structure_levels or word in self.extra_nodes

    @property
    def use_words(self):
        return WORD in self.context_structure_levels

    @property
    def use_tokens(self):
        return TOKEN in self.context_structure_levels