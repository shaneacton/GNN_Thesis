from Code.Config.config import Config


class VisualisationConfig(Config):
    def __init__(self):
        super().__init__()

        self.visualise_graphs = True
        self.exit_after_first_viz = True
        self.max_context_graph_chars = 200  # 400
        self.max_candidates = 3
