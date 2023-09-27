# Dataset loading functionality

import itertools

import torch
from torch.utils.data import IterableDataset, random_split
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset, PPI
from torch_geometric.transforms import AddRandomWalkPE, Compose, ToDevice

from data.generated import construct_generator, GeneratedDataset, GeneratedIterableDataset
from data.transforms import AddLaplacianEigenvectorPE, ToComplement

generated_datasets = [
  'ErdosRenyi',
  'BarabasiAlbert',
  'PowerlawCluster',
  'WattsStrogatz',
  'ForcedRB',
]

TU_datasets = [
  'ENZYMES',
  'PROTEINS',
  'IMDB-BINARY',
  'MUTAG',
  'COLLAB',
  'REDDIT-MULTI-5K',
  'REDDIT-MULTI-12K',
  'REDDIT-BINARY',
]

def construct_dataset(args):
    # precompute laplacian eigenvectors unconditionally
    pre_transforms = []
    if not args.infinite and args.dataset in ['ENZYMES', 'PROTEINS', 'IMDB-BINARY', 'MUTAG', 'COLLAB']:
        pre_transforms.append(AddLaplacianEigenvectorPE(k=8, is_undirected=True))
    elif not args.infinite:
        pre_transforms.append(AddRandomWalkPE(walk_length=8))

    if len(pre_transforms) > 0:
        pre_transform = Compose(pre_transforms)
    else:
        pre_transform = None

    transform = None
    if args.positional_encoding == 'laplacian_eigenvector':
        assert args.pe_dimension <= args.rank
        assert args.pe_dimension <= 8 # for now, this is our maximum
    elif args.positional_encoding == 'random_walk':
        assert args.pe_dimension < args.rank
        #transform = AddRandomWalkPE(walk_length=args.pe_dimension)
        # XXX we implement in featurize_batch in model/training.py
        if args.pe_dimension == 16:
            print("skipping slow, large runs with 16-dimensional RWPE!")
            exit(0)
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

    if args.dataset in generated_datasets:
        generator, name = construct_generator(args)
        if not args.infinite:
            dataset = GeneratedDataset(root=f'{data_root}/generated',
                        name=name,
                        generator=generator,
                        num_graphs=args.num_graphs,
                        seed=args.data_seed,
                        parallel=args.parallel,
                        pre_transform=pre_transform,
                        transform=transform)
        else:
            dataset = GeneratedIterableDataset(
                        generator=generator,
                        seed=args.data_seed,
                        pre_transform=pre_transform,
                        transform=transform)
    elif args.dataset in TU_datasets:
        dataset = TUDataset(root=f'{data_root}',
                    name=args.dataset,
                    pre_transform=pre_transform,
                    transform=transform)
    elif args.dataset == 'PPI':
        # TODO they handle the split differently for this one. we're fetching the training split
        dataset = PPI(root=f'{data_root}',
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
            # TODO make the validation set size controllable from args
            # TODO make the test_loader
            val_loader = DataLoader(list(itertools.islice(dataset, 1000)), batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(list(itertools.islice(dataset, 1000)), batch_size=args.batch_size, shuffle=False)
            return train_loader, val_loader, test_loader
        else:
            raise ValueError(f"Invalid mode passed into construct_loaders: {mode}")

    # default split mode
    if mode == "test":
        # the whole dataset is your loader.
        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        return test_loader
    elif mode is None:
        print("dataset size:", len(dataset))
        train_size = int(args.train_fraction * len(dataset))
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
