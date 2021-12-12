import torch
from torch_geometric.data import InMemoryDataset


class DistancingDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    # @property
    # def raw_file_names(self):
    #     return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    # def process(self):
    #     # Read data into huge `Data` list.
    #     data_list = [...]

    #     if self.pre_filter is not None:
    #         data_list = [data for data in data_list if self.pre_filter(data)]

    #     if self.pre_transform is not None:
    #         data_list = [self.pre_transform(data) for data in data_list]

    #     data, slices = self.collate(data_list)
    #     torch.save((data, slices), self.processed_paths[0])

if __name__ == "__main__":
    dataset = DistancingDataset(root="/home/santapo/OnlineLab/challenges/5k_compliance_zalo/distancing_compliance_gnn/data/distancing")
    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')