from Code.Config.config import Config


class VisualisationConfig(Config):
    def __init__(self):
        super().__init__()

        self.max_context_graph_chars = 30