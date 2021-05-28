
"""aimed to test the way compute resources scale with sparsity for both transformers and GNNs"""
import time

from torch.autograd import profiler
from torch.nn import TransformerEncoderLayer
from torch_geometric.nn import GATConv, SAGEConv

from Code.GNNs.custom_gat import CustomGAT
from Code.Sparsity.custom_transformer import CustomTransformer
from Code.Sparsity.graph_generator import get_dense_graph, get_sparse_graph
from Code.Training import dev

from pytorch_memlab import MemReporter


num_nodes = 300
channels = 400
heads = 5

# gnn = SAGEConv(channels, channels).to(dev())
gnn = CustomGAT(channels, channels, heads, output_linear=True, residual_attention=True, add_self_loops=False).to(dev())
# gnn = GATConv(channels, channels, heads).to(dev())

# transformer = CustomTransformer(channels, heads, dim_feedforward=channels).to(dev())

# nodes, edge_index, mask = get_sparse_graph(num_nodes, channels, 0.01)
nodes, edge_index, mask = get_dense_graph(num_nodes, channels)
print("num nodes:", num_nodes, "num edges:", edge_index.size(1))
# _ = transformer(nodes.view(nodes.size(0), 1, -1), src_mask=mask)
_ = gnn(nodes, edge_index=edge_index)


reporter = MemReporter(gnn)
start_t = time.time()
_ = gnn(nodes, edge_index=edge_index)
gnn_time = time.time()
print("gnn time:", (gnn_time-start_t))

# _ = transformer(nodes.view(nodes.size(0), 1, -1), src_mask=mask)
# transformer_time = time.time()
# print("trans time:", (transformer_time-start_t))
reporter.report()

