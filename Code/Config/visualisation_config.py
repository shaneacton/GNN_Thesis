from Code.Config.config import Config


class VisualisationConfig(Config):
    def __init__(self):
        super().__init__()

        self.visualise_graphs = False
        self.exit_after_first_viz = False
        self.max_context_graph_chars = 250  # 400
        self.max_candidates = 3
