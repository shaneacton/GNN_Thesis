from typing import List
from Code.Config.config import Config


class ConfigSet(Config):

    def __init__(self, configs: List[Config] = None, config: Config = None):
        super().__init__()
        from Code.Config import GNNConfig, GraphEmbeddingConfig, GraphConstructionConfig
        self.gcc: GraphConstructionConfig = None
        self.gec: GraphEmbeddingConfig = None
        self.gnnc: GNNConfig = None
        if config:
            self.add_config(config)
        if configs:
            for config in configs:
                self.add_config(config)

    def get_gnn(self):
        return self.gnnc.get_gnn(self.gcc, self.gec)

    def add_config(self, config: Config):
        from Code.Config import GNNConfig, GraphEmbeddingConfig, GraphConstructionConfig
        if isinstance(config, GraphConstructionConfig):
            if self.gcc:
                raise Exception()
            self.gcc = config
        if isinstance(config, GraphEmbeddingConfig):
            if self.gec:
                raise Exception()
            self.gec = config
        if isinstance(config, GNNConfig):
            if self.gnnc:
                raise Exception()
            self.gnnc = config
