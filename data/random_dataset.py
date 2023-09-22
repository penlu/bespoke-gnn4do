import numpy as np
import networkx as nx

import torch
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx

class RandomGraphDataset(InMemoryDataset):
    def __init__(self, root,
                  num_graphs=10000, num_nodes_per_graph=100, edge_probability=0.15,
                  seed=0, parallel=0,
                  transform=None, pre_transform=None, pre_filter=None):
        self.num_graphs = num_graphs
        self.num_nodes_per_graph = num_nodes_per_graph
        self.edge_probability = edge_probability

        self.seed = seed
        self.parallel = parallel

        super(RandomGraphDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.parallel == 0:
            return [f'random_{self.num_graphs}_{self.num_nodes_per_graph}_{self.edge_probability}_{self.seed}.pt']
        else:
            return [f'random_{self.num_graphs}_{self.num_nodes_per_graph}_{self.edge_probability}_{self.seed}_{self.parallel}.pt']

    def process(self):
        # initialize RNG
        random_state = np.random.RandomState(self.seed)

        # create random undirected graphs and save
        # TODO doing this in parallel when requested
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

class RandomGraphIterableDataset(IterableDataset):
    def __init__(self,
                  num_nodes_per_graph=100, edge_probability=0.15,
                  seed=0,
                  transform=None, pre_transform=None, pre_filter=None):
        self.num_nodes_per_graph = num_nodes_per_graph
        self.edge_probability = edge_probability

        self.seed = seed

        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        super(RandomGraphIterableDataset, self).__init__()

    def __iter__(self):
        # compute our seed
        worker = get_worker_info()
        if worker == None:
            seed = self.seed
        else:
            # if we are a worker, we generate a seed
            # as written this should put out 256 bits: collision is unlikely
            seed = np.random.SeedSequence(entropy=self.seed, spawn_key=(worker.id,)).generate_state(8)

        # initialize RNG
        random_state = np.random.RandomState(seed)

        # generate random graphs forever
        while True:
            G = nx.erdos_renyi_graph(self.num_nodes_per_graph, self.edge_probability, seed=random_state)
            G = from_networkx(G)

            # apply filters and transforms
            if self.pre_filter is not None and not self.pre_filter(G):
                continue
            if self.pre_transform is not None:
                G = self.pre_transform(G)
            if self.transform is not None:
                G = self.transform(G)

            yield G
