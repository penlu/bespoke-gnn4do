import itertools
from itertools import product
import random
import numpy as np
import networkx as nx
import json
import time 
import gurobipy as gp
from gurobipy import GRB

import torch
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx

# Adapted from Nikos's Forced_RB.ipynb

## RB MODEL for hard independent set instances: 
#1) Create n disjoint cliques
#2) Pick any pair of those disjoint cliques and sample s amount of edges between them
#3) Repeat step 2 x iterations
#4) The maximum independent set will be at most n. 
#5) The is_forced argument enforces the size of the maximum independent set to be n. This can slow down the generation process slightly.
#To force the independent set to be exactly n we can create a set of nodes that contains exactly one node from each disjoint clique, and then ensure that no edge gets sampled between them.

def RB_model(generator=np.random.default_rng(0), n_range=[10, 26], k_range=[5, 21], is_forced=True):
    n = generator.integers(n_range[0], n_range[1]) #number of disjoint cliques (and upper bound on max independent set), feel free to change the ranges depending on comp budget or other considerations
    k = generator.integers(k_range[0], k_range[1])  #number of nodes on each disjoint clique, feel free to change as well

    p = generator.uniform(0.3, 1.0) #determines how dense the sampling will be
    a = np.log(k) / np.log(n) #parameter that determines how many edges we sample
    r = - a / np.log(1 - p)
    v = k * n #total number of nodes in the graph
    s = int(p * (n ** (2 * a)))
    iterations = int(r * n * np.log(n) - 1)


    parts = np.reshape(np.int64(range(v)), (n, k))
    total_edges = set()
    for i in parts:
        total_edges |= set(itertools.combinations(i, 2))

    edges = set()
    for _ in range(iterations):
        i, j = generator.choice(n, 2, replace=False)
        all = set(itertools.product(parts[i, :], parts[j, :]))
        all -= edges

        edges |= set(map(tuple, generator.choice(tuple(all), min(s, len(all)), replace=False)))
        total_edges |= edges

    if is_forced: ##this enforces a maximum independent set of size n py picking a random node from each disjoint clique, and then removing all the edges between all the nodes that have been picked. 
        indset = []
        for part in parts:
            indset_node = generator.choice(len(part),1)
            indset +=[part[indset_node].item()]
        forbidden = itertools.combinations(indset, 2)
        for edge in forbidden:
            while edge in total_edges:
                total_edges.remove(edge)
            while tuple(reversed(edge)) in total_edges:
                total_edges.remove(tuple(reversed(edge)))

    G = nx.Graph()
    G.add_edges_from(list(total_edges))
    G = G.to_undirected()
    return G

class ForcedRBDataset(InMemoryDataset):
    def __init__(self, root,
                  num_graphs=1000, n_range=[10, 26], k_range=[5, 21],
                  seed=0, parallel=0,
                  transform=None, pre_transform=None, pre_filter=None):
        self.num_graphs = num_graphs
        self.n_range = n_range
        self.k_range = k_range

        self.seed = seed
        self.parallel = parallel

        super(ForcedRBDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # no files needed
        return []

    @property
    def processed_file_names(self):
        return [f'forcedRB_{self.num_graphs}_{self.seed}_{self.parallel}.pt']

    def download(self):
        # no download needed
        pass

    def process(self):
        # TODO actually handle parallel
        # TODO use SeedSequence and multiprocessing here
        generator = np.random.default_rng(self.seed)

        data_list = []
        for i in range(self.num_graphs):
            G = RB_model(generator=generator)
            data_list.append(from_networkx(G))
            print(f'generated {i+1} of {self.num_graphs}')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class ForcedRBIterableDataset(IterableDataset):
    def __init__(self,
                  num_graphs=1000, n_range=[10, 26], k_range=[5, 21],
                  seed=0,
                  transform=None, pre_transform=None, pre_filter=None):
        self.num_graphs = num_graphs
        self.n_range = n_range
        self.k_range = k_range

        self.seed = seed

        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        super(ForcedRBIterableDataset, self).__init__()

    def __iter__(self):
        # compute our seed
        worker = get_worker_info()
        if worker == None:
            seed = self.seed
        else:
            # if we are a worker, we generate a seed
            # as written this should put out 256 bits: collision is unlikely
            seed = np.random.SeedSequence(entropy=self.seed, spawn_key=(worker.id,))

        # initialize RNG
        generator = np.random.default_rng(seed)

        # generate random graphs forever
        while True:
            G = RB_model(generator=generator)
            G = from_networkx(G)

            # apply filters and transforms
            if self.pre_filter is not None and not self.pre_filter(G):
                continue
            if self.pre_transform is not None:
                G = self.pre_transform(G)
            if self.transform is not None:
                G = self.transform(G)

            yield G
