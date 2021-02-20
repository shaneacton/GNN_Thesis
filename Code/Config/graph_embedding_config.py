from Code.Config.config import Config


class GraphEmbeddingConfig(Config):

    def __init__(self):
        super().__init__()
        self.max_pad_volume = 50000