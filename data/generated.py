import itertools
import numpy as np
import networkx as nx

import torch
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx

from data.forced_rb_dataset import forced_rb_generator

def construct_generator(args):
    if args.dataset == 'ErdosRenyi':
        generator = lambda seed: erdos_renyi_generator(seed, n=args.gen_n, p=args.gen_p)
        name = f'erdos_renyi_n{args.gen_n}_p{args.gen_p}'
    elif args.dataset == 'BarabasiAlbert':
        generator = lambda seed: barabasi_albert_generator(seed, n=args.gen_n, m=args.gen_m)
        name = f'barabasi_albert_n{args.gen_n}_m{args.gen_m}'
    elif args.dataset == 'PowerlawCluster':
        generator = lambda seed: powerlaw_cluster_generator(seed, n=args.gen_n, m=args.gen_m, p=args.gen_p)
        name = f'powerlaw_cluster_n{args.gen_n}_m{args.gen_m}_p{args.gen_p}'
    elif args.dataset == 'WattsStrogatz':
        generator = lambda seed: watts_strogatz_generator(seed, n=args.gen_n, k=args.gen_k, p=args.gen_p)
        name = f'watts_strogatz_n{args.gen_n}_k{args.gen_k}_p{args.gen_p}'
    elif args.dataset == 'ForcedRB':
        generator = lambda seed: forced_rb_generator(seed, n=args.RB_n, k=args.RB_k)
        name = f'forced_rb_n{args.RB_n}_k{args.RB_k}'
    else:
        raise ValueError('Got a bad generated dataset: {args.dataset}')

    return generator, name

def erdos_renyi_generator(seed, n=100, p=0.15):
    if isinstance(n, int):
        n_min = n
        n_max = n
    elif isinstance(n, list) and len(n) == 1:
        n_min = n[0]
        n_max = n[0]
    elif isinstance(n, list) and len(n) == 2:
        n_min, n_max = n
    else:
        raise ValueError('erdos_renyi_generator got bad n (expected int, [n], or [n_min, n_max]: {n}')

    random_state = np.random.RandomState(seed)
    while True:
        if n_min != n_max:
            n = random_state.randint(n_min, n_max + 1)
        else:
            n = n_min
        G = nx.erdos_renyi_graph(n, p, seed=random_state)
        yield from_networkx(G)

def barabasi_albert_generator(seed, n=100, m=4):
    if isinstance(n, int):
        n_min = n
        n_max = n
    elif isinstance(n, list) and len(n) == 1:
        n_min = n[0]
        n_max = n[0]
    elif isinstance(n, list) and len(n) == 2:
        n_min, n_max = n
    else:
        raise ValueError('barabasi_albert_generator got bad n (expected int, [n], or [n_min, n_max]: {n}')

    random_state = np.random.RandomState(seed)
    while True:
        if n_min != n_max:
            n = random_state.randint(n_min, n_max + 1)
        else:
            n = n_min
        G = nx.barabasi_albert_graph(n, m, seed=random_state)
        yield from_networkx(G)

def powerlaw_cluster_generator(seed, n=100, m=4, p=0.25):
    if isinstance(n, int):
        n_min = n
        n_max = n
    elif isinstance(n, list) and len(n) == 1:
        n_min = n[0]
        n_max = n[0]
    elif isinstance(n, list) and len(n) == 2:
        n_min, n_max = n
    else:
        raise ValueError('powerlaw_cluster_generator got bad n (expected int, [n], or [n_min, n_max]: {n}')

    random_state = np.random.RandomState(seed)
    while True:
        if n_min != n_max:
            n = random_state.randint(n_min, n_max + 1)
        else:
            n = n_min
        G = nx.powerlaw_cluster_graph(n, m, p, seed=random_state)
        yield from_networkx(G)

def watts_strogatz_generator(seed, n=100, k=4, p=0.25):
    if isinstance(n, int):
        n_min = n
        n_max = n
    elif isinstance(n, list) and len(n) == 1:
        n_min = n[0]
        n_max = n[0]
    elif isinstance(n, list) and len(n) == 2:
        n_min, n_max = n
    else:
        raise ValueError('watts_strogatz_generator got bad n (expected int, [n], or [n_min, n_max]: {n}')

    random_state = np.random.RandomState(seed)
    while True:
        if n_min != n_max:
            n = random_state.randint(n_min, n_max + 1)
        else:
            n = n_min
        G = nx.watts_strogatz_graph(n, k, p, seed=random_state)
        yield from_networkx(G)

# TODO chordal graph generation

class GeneratedDataset(InMemoryDataset):
    # generator should be a function that accepts a seed and produces an iterator over data
    def __init__(self, root, name, generator,
                  num_graphs=1000, seed=0, parallel=0,
                  transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.generator = generator
        self.num_graphs = num_graphs
        self.seed = seed
        self.parallel = parallel

        super(GeneratedDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'{self.name}_seed{self.seed}_par{self.parallel}_N{self.num_graphs}.pt']

    def process(self):
        # initialize RNG
        worker = get_worker_info()
        if worker == None:
            seed = self.seed
        else:
            # if we are a worker, we generate a seed
            # as written this should put out 256 bits: collision is unlikely
            seed = np.random.SeedSequence(entropy=self.seed, spawn_key=(worker.id,)).generate_state(8)

        # generate graphs and save
        # TODO doing this in parallel when requested
        data_list = list(itertools.islice(self.generator(seed), self.num_graphs))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class GeneratedIterableDataset(IterableDataset):
    def __init__(self, generator, seed=0,
                  transform=None, pre_transform=None, pre_filter=None):
        self.generator = generator
        self.seed = seed

        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        super(GeneratedIterableDataset, self).__init__()

    def __iter__(self):
        # compute our seed
        worker = get_worker_info()
        if worker == None:
            seed = self.seed
        else:
            # if we are a worker, we generate a seed
            # as written this should put out 256 bits: collision is unlikely
            seed = np.random.SeedSequence(entropy=self.seed, spawn_key=(worker.id,)).generate_state(8)

        # generate graphs forever
        for G in self.generator(seed):
            # apply filters and transforms
            if self.pre_filter is not None and not self.pre_filter(G):
                continue
            if self.pre_transform is not None:
                G = self.pre_transform(G)
            if self.transform is not None:
                G = self.transform(G)

            yield G
