from Code.Config.config_set import ConfigSet
from Code.Config.evaluation_config import EvaluationConfig
from Code.Config.gnn_config import GNNConfig
from Code.Config.graph_construction_config import GraphConstructionConfig
from Code.Config.graph_embedding_config import GraphEmbeddingConfig
from Code.Config.system_config import SystemConfig

gcc = GraphConstructionConfig()
gec = GraphEmbeddingConfig()
gnnc = GNNConfig()

configs = ConfigSet(configs=[gcc, gec, gnnc])

eval_conf = EvaluationConfig()
sysconf = SystemConfig()