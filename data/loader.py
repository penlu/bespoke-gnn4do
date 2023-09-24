# Dataset loading functionality

import itertools

import torch
from torch.utils.data import IterableDataset, random_split
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import AddRandomWalkPE, Compose

from data.forced_rb_dataset import ForcedRBDataset, ForcedRBIterableDataset
from data.random_dataset import RandomGraphDataset, RandomGraphIterableDataset
from data.random_dataset import BarabasiAlbertDataset, BarabasiAlbertIterableDataset
from data.random_dataset import PowerlawClusterDataset, PowerlawClusterIterableDataset
from data.random_dataset import WattsStrogatzDataset, WattsStrogatzIterableDataset
from data.transforms import AddLaplacianEigenvectorPE, ToComplement

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
        raise ValueError(f"Invalid positional encoding passed into construct_dataset: {args.positional_encoding}")

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
                    seed=args.data_seed,
                    parallel=args.parallel,
                    pre_transform=pre_transform,
                    transform=transform)
    elif args.dataset == 'RANDOM_inf':
        dataset = RandomGraphIterableDataset(
                    num_nodes_per_graph=args.num_nodes_per_graph,
                    edge_probability=args.edge_probability,
                    seed=args.data_seed,
                    pre_transform=pre_transform,
                    transform=transform)
    elif args.dataset == 'ForcedRB':
        dataset = ForcedRBDataset(root=f'{data_root}/forced_rb',
                    num_graphs=args.num_graphs,
                    n_range=args.RB_n,
                    k_range=args.RB_k,
                    seed=args.data_seed,
                    parallel=args.parallel,
                    pre_transform=pre_transform,
                    transform=transform)
    elif args.dataset == 'ForcedRB_inf':
        dataset = ForcedRBIterableDataset(
                    n_range=args.RB_n,
                    k_range=args.RB_k,
                    seed=args.data_seed,
                    pre_transform=pre_transform,
                    transform=transform)
    # TODO chordal graph generation
    elif args.dataset == 'TU':
        dataset = TUDataset(root=f'{data_root}',
                    name=args.TUdataset_name,
                    pre_transform=pre_transform,
                    transform=transform)
    else:
        raise ValueError(f"Invalid dataset passed into construct_dataset: {args.dataset}")

    return dataset

def construct_loaders(args, mode=None):
    ''' dataloader construction
    
    constructs train and val loaders if mode = None
    constructs test loader if mode = Test

    '''
    dataset = construct_dataset(args)

    if isinstance(dataset, IterableDataset):
        # infinite data training
        train_loader = DataLoader(dataset,
            batch_size=args.batch_size,
            num_workers=args.parallel,
        )

        if mode == "test":
            return train_loader
        elif mode is None:
            # TODO what to do about the validation set when infinite data?
            # TODO make the validation set size controllable
            # TODO make the test_loader
            val_loader = DataLoader(list(itertools.islice(dataset, 100)), batch_size=args.batch_size, shuffle=False)
            return train_loader, val_loader, None
        else:
            raise ValueError(f"Invalid mode passed into construct_loaders: {mode}")

    # default split mode
    if mode == "test":
        # the whole dataset is your loader.
        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        return test_loader
    elif mode is None:
        # TODO make this depend on args for split size
        print("dataset size:", len(dataset))
        train_size = int(0.8 * len(dataset))
        val_size = (len(dataset) - train_size)//2
        test_size = len(dataset) - train_size - val_size

        generator = torch.Generator().manual_seed(args.split_seed)
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        
        return train_loader, val_loader, test_loader
    else:
        raise ValueError(f"Invalid mode passed into construct_loaders: {mode}")
