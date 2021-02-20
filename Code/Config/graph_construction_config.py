from Code.Config.config import Config
from Code.constants import ENTITY, COREF, TOKEN, WORD, SENTENCE, PARAGRAPH, DOCUMENT, CONNECTION_TYPE, WINDOW, \
    SEQUENTIAL, WINDOW_SIZE, NOUN, CONTEXT, QUERY


class GraphConstructionConfig(Config):

    def __init__(self):

        super().__init__()
        self.max_edges = 400000  # 400000
        # cuts off all chars after the max. tries to proceed with partial context. -1 to turn off
        self.max_context_chars = -1

        if self.max_context_chars != -1:
            print("truncating contexts to", self.max_context_chars, "chars")