import torch
from torch_geometric.data import Batch, Data

from Code.Data.Graph.Types.type_map import TypeMap
from Code.Training import device

batch_size = 3


example_x = torch.tensor([[2, 1, 3], [5, 6, 4], [3, 7, 5], [12, 0, 6]], dtype=torch.float).to(device)
example_y = torch.tensor([0, 1, 0, 1], dtype=torch.float).to(device)

example_edge_index = torch.tensor([[0, 2, 1, 0, 3],
                                   [3, 1, 0, 1, 2]], dtype=torch.long).to(device)

example_edge_types = torch.tensor([[0, 2, 1, 0, 3]], dtype=torch.long).to(device)

example_data = Data(x=example_x, y=example_y, edge_index=example_edge_index, edge_types=example_edge_types).to(device)


example_batch = Batch.from_data_list([example_data] * batch_size).to(device)
example_batch.edge_types = example_batch.edge_types.view(-1)