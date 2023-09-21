import numpy as np
import networkx as nx

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx

class RandomGraphDataset(InMemoryDataset):
    def __init__(self, root,
                  num_graphs=10000, num_nodes_per_graph=100, edge_probability=0.15,
                  seed=0,
                  transform=None, pre_transform=None, pre_filter=None):
        self.num_graphs = num_graphs
        self.num_nodes_per_graph = num_nodes_per_graph
        self.edge_probability = edge_probability
        self.seed = seed
        super(RandomGraphDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # no files needed
        return []

    @property
    def processed_file_names(self):
        return [f'random_{self.num_graphs}_{self.num_nodes_per_graph}_{self.edge_probability}_{self.seed}.pt']

    def download(self):
        # no download needed
        pass

    def process(self):
        # create random undirected graphs and save
        random_state = np.random.RandomState(self.seed)
        data_list = []
        for i in range(self.num_graphs):
            G = nx.erdos_renyi_graph(self.num_nodes_per_graph, self.edge_probability, seed=random_state)
            data_list.append(from_networkx(G))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

