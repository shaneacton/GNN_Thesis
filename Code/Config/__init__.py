from Code.Config.config_set import ConfigSet
from Code.Config.gnn_config import GNNConfig
from Code.Config.graph_construction_config import GraphConstructionConfig
from Code.Config.graph_embedding_config import GraphEmbeddingConfig

gcc = GraphConstructionConfig()
gec = GraphEmbeddingConfig()
gnnc = GNNConfig()

configs = ConfigSet(configs=[gcc, gec, gnnc])
