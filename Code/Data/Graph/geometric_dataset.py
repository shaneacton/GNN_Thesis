import torch
from torch_geometric.data import InMemoryDataset


class MyOwnDataset(InMemoryDataset):

    """

    """

    def __init__(self, root):
        super(MyOwnDataset, self).__init__(root, None, None)
        # process is called by super
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['geometric_data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = [...]


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])