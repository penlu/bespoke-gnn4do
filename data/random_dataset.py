import numpy as np
import networkx as nx
from torch_geometric.utils.convert import from_networkx

from data.generated import GeneratedDataset, GeneratedIterableDataset

# TODO XXX make all the num_nodes into ranges
class RandomGraphDataset(GeneratedDataset):
    def __init__(self, root,
                  num_nodes_per_graph=100, edge_probability=0.15,
                  **kwargs):
        self.num_nodes_per_graph = num_nodes_per_graph
        self.edge_probability = edge_probability
        super(RandomGraphDataset, self).__init__(root, **kwargs)

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
    def __init__(self,
                  num_nodes_per_graph=100, edge_probability=0.15,
                  **kwargs):
        self.num_nodes_per_graph = num_nodes_per_graph
        self.edge_probability = edge_probability
        super(RandomGraphIterableDataset, self).__init__(**kwargs)

    def generate(self, seed, **kwargs):
        random_state = np.random.RandomState(self.seed)
        while True:
            G = nx.erdos_renyi_graph(self.num_nodes_per_graph, self.edge_probability, seed=random_state)
            yield from_networkx(G)

class BarabasiAlbertDataset(GeneratedDataset):
    def __init__(self, root, n=100, m=4, **kwargs):
        self.n = n
        self.m = m
        super(BarabasiAlbertDataset, self).__init__(root, **kwargs)

    @property
    def processed_file_names(self):
        if self.parallel == 0:
            return [f'barabasi_albert_{self.num_graphs}_{self.n}_{self.m}_{self.seed}.pt']
        else:
            return [f'barabasi_albert_{self.num_graphs}_{self.n}_{self.m}_{self.seed}_{self.parallel}.pt']

    def generate(self, seed, **kwargs):
        random_state = np.random.RandomState(self.seed)
        while True:
            G = nx.barabasi_albert_graph(self.n, self.m, seed=random_state)
            yield from_networkx(G)

class BarabasiAlbertIterableDataset(GeneratedIterableDataset):
    def __init__(self, n=100, m=4, **kwargs):
        self.n = n
        self.m = m
        super(BarabasiAlbertIterableDataset, self).__init__(**kwargs)

    def generate(self, seed, **kwargs):
        random_state = np.random.RandomState(self.seed)
        while True:
            G = nx.barabasi_albert_graph(self.n, self.m, seed=random_state)
            yield from_networkx(G)

class PowerlawClusterDataset(GeneratedDataset):
    def __init__(self, root, n=100, m=4, p=0.25, **kwargs):
        self.n = n
        self.m = m
        self.p = p
        super(PowerlawClusterDataset, self).__init__(root, **kwargs)

    @property
    def processed_file_names(self):
        if self.parallel == 0:
            return [f'powerlaw_cluster_{self.num_graphs}_{self.n}_{self.m}_{self.p}_{self.seed}.pt']
        else:
            return [f'powerlaw_cluster_{self.num_graphs}_{self.n}_{self.m}_{self.p}_{self.seed}_{self.parallel}.pt']

    def generate(self, seed, **kwargs):
        random_state = np.random.RandomState(self.seed)
        while True:
            G = nx.powerlaw_cluster_graph(self.n, self.m, self.p, seed=random_state)
            yield from_networkx(G)

class PowerlawClusterIterableDataset(GeneratedIterableDataset):
    def __init__(self, n=100, m=4, p=0.25, **kwargs):
        self.n = n
        self.m = m
        self.p = p
        super(PowerlawClusterIterableDataset, self).__init__(**kwargs)

    def generate(self, seed, **kwargs):
        random_state = np.random.RandomState(self.seed)
        while True:
            G = nx.powerlaw_cluster_graph(self.n, self.m, self.p, seed=random_state)
            yield from_networkx(G)

class WattsStrogatzDataset(GeneratedDataset):
    def __init__(self, root, n=100, k=4, p=0.25, **kwargs):
        self.n = n
        self.k = k
        self.p = p
        super(WattsStrogatzDataset, self).__init__(root, **kwargs)

    @property
    def processed_file_names(self):
        if self.parallel == 0:
            return [f'watts_strogatz_cluster_{self.num_graphs}_{self.n}_{self.k}_{self.p}_{self.seed}.pt']
        else:
            return [f'watts_strogatz_cluster_{self.num_graphs}_{self.n}_{self.k}_{self.p}_{self.seed}_{self.parallel}.pt']

    def generate(self, seed, **kwargs):
        random_state = np.random.RandomState(self.seed)
        while True:
            G = nx.watts_strogatz_graph(self.n, self.k, self.p, seed=random_state)
            yield from_networkx(G)

class WattsStrogatzIterableDataset(GeneratedIterableDataset):
    def __init__(self, n=100, k=4, p=0.25, **kwargs):
        self.n = n
        self.k = k
        self.p = p
        super(WattsStrogatzDataset, self).__init__(**kwargs)

    def generate(self, seed, **kwargs):
        random_state = np.random.RandomState(self.seed)
        while True:
            G = nx.watts_strogatz_graph(self.n, self.k, self.p, seed=random_state)
            yield from_networkx(G)
