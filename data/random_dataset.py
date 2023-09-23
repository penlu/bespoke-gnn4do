import numpy as np
import networkx as nx

import torch
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx

from data.generated import GeneratedDataset, GeneratedIterableDataset

class RandomGraphDataset(GeneratedDataset):
    def __init__(self, root, num_graphs=1000, seed=0, parallel=0,
                  num_nodes_per_graph=100, edge_probability=0.15,
                  transform=None, pre_transform=None, pre_filter=None):
        self.num_nodes_per_graph = num_nodes_per_graph
        self.edge_probability = edge_probability

        super(RandomGraphDataset, self).__init__(
            root, num_graphs=num_graphs, seed=seed, parallel=parallel,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter)

    @property
    def processed_file_names(self):
        if self.parallel == 0:
            return [f'random_{self.num_graphs}_{self.num_nodes_per_graph}_{self.edge_probability}_{self.seed}.pt']
        else:
            return [f'random_{self.num_graphs}_{self.num_nodes_per_graph}_{self.edge_probability}_{self.seed}_{self.parallel}.pt']

    def generate(self, seed, **kwargs):
        random_state = np.random.RandomState(self.seed)
        while True:
            G = nx.erdos_renyi_graph(self.num_nodes_per_graph, self.edge_probability, seed=random_state)
            yield from_networkx(G)

class RandomGraphIterableDataset(GeneratedIterableDataset):
    def __init__(self, seed=0,
                  num_nodes_per_graph=100, edge_probability=0.15,
                  transform=None, pre_transform=None, pre_filter=None):
        self.num_nodes_per_graph = num_nodes_per_graph
        self.edge_probability = edge_probability

        super(RandomGraphIterableDataset, self).__init__(seed=seed,
            transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

    def generate(self, seed, **kwargs):
        random_state = np.random.RandomState(self.seed)
        while True:
            G = nx.erdos_renyi_graph(self.num_nodes_per_graph, self.edge_probability, seed=random_state)
            yield from_networkx(G)
