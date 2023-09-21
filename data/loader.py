# Dataset loading functionality

import torch
from torch.utils.data import random_split
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils.convert import from_networkx
from torch_geometric.transforms import AddRandomWalkPE, Compose

from data.forced_rb import RB_model
from data.transforms import AddLaplacianEigenvectorPE, ToComplement

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
            G = networkx.erdos_renyi_graph(self.num_nodes_per_graph, self.edge_probability, seed=random_state)
            data_list.append(from_networkx(G))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class ForcedRBDataset(InMemoryDataset):
    def __init__(self, root,
                  num_graphs=1000,
                  seed=0, parallel=8,
                  transform=None, pre_transform=None, pre_filter=None):
        self.num_graphs = num_graphs
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
        # create random undirected graphs and save
        generator = np.random.default_rng(self.seed)

        # TODO use SeedSequence and multiprocessing here

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

def construct_dataset(args):
    # precompute laplacian eigenvectors unconditionally
    pre_transform = AddLaplacianEigenvectorPE(k=8, is_undirected=True)

    transform = None
    if args.positional_encoding == 'laplacian_eigenvector':
        assert args.pe_dimension <= args.rank
        assert args.pe_dimension <= 8 # for now, this is our maximum
    elif args.positional_encoding == 'random_walk':
        assert args.pe_dimension < args.rank
        transform = AddRandomWalkPE(walk_length=args.pe_dimension)
    elif args.positional_encoding is not None:
        raise ValueError(f"Invalid positional encoding passed into construct_loaders: {args.positional_encoding}")

    # we do max clique by running VC on graph complement
    # XXX kinda grody!!!
    if args.problem_type == 'max_clique':
        if pre_transform is not None:
            pre_transform = Compose([ToComplement(), pre_transform])
        else:
            pre_transform = ToComplement()
        data_root = 'datasets_complement'
    else:
        data_root = 'datasets'

    if args.dataset == 'RANDOM':
        dataset = RandomGraphDataset(root=f'{data_root}/random',
                    num_graphs=args.num_graphs,
                    num_nodes_per_graph=args.num_nodes_per_graph,
                    edge_probability=args.edge_probability,
                    pre_transform=pre_transform,
                    transform=transform)
    elif args.dataset == 'TU':
        dataset = TUDataset(root=f'{data_root}',
                    name=args.TUdataset_name,
                    pre_transform=pre_transform,
                    transform=transform)
    elif args.dataset == 'ForcedRB':
        # TODO other parameters for graph generation
        dataset = ForcedRBDataset(root=f'{data_root}/forced_rb',
                    num_graphs=args.num_graphs,
                    pre_transform=pre_transform,
                    transform=transform)
    else:
        raise ValueError(f"Unimplemented dataset {args.dataset}. Expected RANDOM or TU.")

    return dataset

def construct_loaders(args, mode=None):
    ''' dataloader construction
    
    constructs train and val loaders if mode = None
    constructs test loader if mode = Test

    '''
    dataset = construct_dataset(args)

    if mode == "test":
        # the whole dataset is your loader.
        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        return test_loader
    elif mode is None:
        # TODO make this depend on args for split size
        print("dataset size:", len(dataset))
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        return train_loader, val_loader
    else:
        raise ValueError(f"Invalid mode passed into construct_loaders: {mode}")
