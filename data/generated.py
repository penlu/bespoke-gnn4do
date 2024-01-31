import itertools
import numpy as np
import networkx as nx

import torch
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx

from data.forced_rb_dataset import forced_rb_generator
from data.sat import random_3sat_generator

# some of our command line arguments can be int, [int], or [int, int]
# disassemble them into a min and max
# XXX this possibly should just become the default action
def disassemble_param(n, default=100):
    if n is None:
        if isinstance(default, int) or isinstance(default, float):
            n_min = default
            n_max = default
        else:
            n_min = default[0]
            n_max = default[1]
    elif isinstance(n, int):
        n_min = n
        n_max = n
    elif isinstance(n, list) and len(n) == 1:
        n_min = n[0]
        n_max = n[0]
    elif isinstance(n, list) and len(n) == 2:
        n_min, n_max = n
    else:
        raise ValueError('construct_generator got bad arg (expected int, [int], or [int, int]: {n}')

    return n_min, n_max

def format_param(n_min, n_max):
    if n_min == n_max:
        return str(n_min)
    else:
        return str(n_min) + ',' + str(n_max)

# a generator is a function that takes a seed and produces an iterator
def construct_generator(args):
    if args.dataset == 'ErdosRenyi':
        n_min, n_max = disassemble_param(args.gen_n, default=100)
        p_min, p_max = disassemble_param(args.gen_p, default=0.15)
        generator = lambda seed: erdos_renyi_generator(seed, n_min, n_max, p_min, p_max)
        name = f'erdos_renyi_n{format_param(n_min, n_max)}_p{format_param(p_min, p_max)}'
    elif args.dataset == 'BarabasiAlbert':
        n_min, n_max = disassemble_param(args.gen_n, default=100)
        m_min, m_max = disassemble_param(args.gen_m, default=4)
        generator = lambda seed: barabasi_albert_generator(seed, n_min, n_max, m_min, m_max)
        name = f'barabasi_albert_n{format_param(n_min, n_max)}_m{format_param(m_min, m_max)}'
    elif args.dataset == 'PowerlawCluster':
        n_min, n_max = disassemble_param(args.gen_n, default=100)
        m_min, m_max = disassemble_param(args.gen_m, default=4)
        p_min, p_max = disassemble_param(args.gen_p, default=0.25)
        generator = lambda seed: powerlaw_cluster_generator(seed, n_min, n_max, m_min, m_max, p_min, p_max)
        name = f'powerlaw_cluster_n{format_param(n_min, n_max)}_m{format_param(m_min, m_max)}_p{format_param(p_min, p_max)}'
    elif args.dataset == 'WattsStrogatz':
        n_min, n_max = disassemble_param(args.gen_n, default=100)
        k_min, k_max = disassemble_param(args.gen_k, default=4)
        p_min, p_max = disassemble_param(args.gen_p, default=0.25)
        generator = lambda seed: watts_strogatz_generator(seed, n_min, n_max, k_min, k_max, p_min, p_max)
        name = f'watts_strogatz_n{format_param(n_min, n_max)}_k{format_param(k_min, k_max)}_p{format_param(p_min, p_max)}'
    elif args.dataset == 'ForcedRB':
        n_min, n_max = disassemble_param(args.gen_n, default=(10, 26))
        k_min, k_max = disassemble_param(args.gen_k, default=(5, 21))
        p_min, p_max = disassemble_param(args.gen_p, default=(0.3, 1.0))
        generator = lambda seed: forced_rb_generator(seed, n_min, n_max, k_min, k_max, p_min, p_max)
        name = f'forced_rb_n{format_param(n_min, n_max)}_k{format_param(k_min, k_max)}'
    elif args.dataset == 'random-sat':
        n_min, n_max = disassemble_param(args.gen_n, default=100)
        k_min, k_max = disassemble_param(args.gen_k, default=400)
        p_min, p_max = disassemble_param(args.gen_p, default=0.5)
        generator = lambda seed: random_3sat_generator(seed, n_min, n_max, k_min, k_max, p_min, p_max)
        name = f'sat_n{format_param(n_min, n_max)}_k{format_param(k_min, k_max)}_p{format_param(p_min, p_max)}'
    else:
        raise ValueError('Got a bad generated dataset: {args.dataset}')

    return generator, name

# invoke RNG to generate random int only when a non-empty range is supplied
def conditional_randint(random_state, n_min, n_max):
    if n_min != n_max:
        n = random_state.randint(n_min, n_max + 1)
    else:
        n = n_min
    return n

# invoke RNG to generate random float only when a non-empty range is supplied
def conditional_rand(random_state, n_min, n_max):
    if n_min != n_max:
        n = random_state.uniform(n_min, n_max)
    else:
        n = n_min
    return n

def erdos_renyi_generator(seed, n_min=100, n_max=100, p_min=0.15, p_max=0.15):
    random_state = np.random.RandomState(seed)
    while True:
        n = conditional_randint(random_state, n_min, n_max)
        p = conditional_rand(random_state, p_min, p_max)
        G = nx.erdos_renyi_graph(n, p, seed=random_state)
        yield from_networkx(G)

def barabasi_albert_generator(seed, n_min=100, n_max=100, m_min=4, m_max=4):
    random_state = np.random.RandomState(seed)
    while True:
        n = conditional_randint(random_state, n_min, n_max)
        m = conditional_randint(random_state, m_min, m_max)
        G = nx.barabasi_albert_graph(n, m, seed=random_state)
        yield from_networkx(G)

def powerlaw_cluster_generator(seed, n_min=100, n_max=100, m_min=4, m_max=4, p_min=0.25, p_max=0.25):
    random_state = np.random.RandomState(seed)
    while True:
        n = conditional_randint(random_state, n_min, n_max)
        m = conditional_randint(random_state, m_min, m_max)
        p = conditional_rand(random_state, p_min, p_max)
        G = nx.powerlaw_cluster_graph(n, m, p, seed=random_state)
        yield from_networkx(G)

def watts_strogatz_generator(seed, n_min=100, n_max=100, k_min=4, k_max=4, p_min=0.25, p_max=0.25):
    random_state = np.random.RandomState(seed)
    while True:
        n = conditional_randint(random_state, n_min, n_max)
        k = conditional_randint(random_state, k_min, k_max)
        p = conditional_rand(random_state, p_min, p_max)
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
