import torch
from torch_geometric.data import Data, Batch

edge_index_s = torch.tensor([
    [0, 0, 0, 0],
    [1, 2, 3, 4],
])
x_s = torch.randn(5, 16)  # 5 nodes.
edge_index_t = torch.tensor([
    [0, 0, 0],
    [1, 2, 3],
])
x_t = torch.randn(4, 16)  # 4 nodes.

d1 = Data(x_s, edge_index_s)
d2 = Data(x_t, edge_index_t)
print("d1:", d1, "\nd2:", d2)

batch = Batch.from_data_list([d1, d2])
print("batch:", batch)

print("b_x:", batch.x)
print("b_e:", batch.edge_index)
