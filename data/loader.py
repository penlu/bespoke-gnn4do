# Dataset loading functionality

from typing import Any, Optional

import networkx
import numpy as np

import torch
from torch.utils.data import random_split
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils.convert import from_networkx
from torch_geometric.transforms import BaseTransform, AddRandomWalkPE
from torch_geometric.utils import (
    get_laplacian,
    get_self_loop_attr,
    scatter,
    to_edge_index,
    to_dense_adj,
    to_torch_csr_tensor,
)

from utils.graph_utils import gen_graph

# cribbed from torch_geometric.transforms, modified to use dense eigenvalue functions
def add_node_attr(data: Data, value: Any,
                  attr_name: Optional[str] = None) -> Data:
    # TODO Move to `BaseTransform`.
    if attr_name is None:
        if 'x' in data:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data

@functional_transform('add_laplacian_eigenvectors')
class AddLaplacianEigenvectorPE(BaseTransform):
    r"""Adds the Laplacian eigenvector positional encoding from the
    `"Benchmarking Graph Neural Networks" <https://arxiv.org/abs/2003.00982>`_
    paper to the given graph
    (functional name: :obj:`add_laplacian_eigenvector_pe`).

    Args:
        k (int): The number of non-trivial eigenvectors to consider.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"laplacian_eigenvector_pe"`)
        is_undirected (bool, optional): If set to :obj:`True`, this transform
            expects undirected graphs as input, and can hence speed up the
            computation of eigenvectors. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :meth:`scipy.linalg.eig` (when :attr:`is_undirected` is
            :obj:`False`) or :meth:`scipy.linalg.eigh` (when
            :attr:`is_undirected` is :obj:`True`).
    """
    def __init__(
        self,
        k: int,
        attr_name: Optional[str] = 'laplacian_eigenvector_pe',
        is_undirected: bool = False,
        **kwargs,
    ):
        self.k = k
        self.attr_name = attr_name
        self.is_undirected = is_undirected
        self.kwargs = kwargs

    def __call__(self, data: Data) -> Data:
        from scipy.linalg import eig, eigh
        eig_fn = eig if not self.is_undirected else eigh

        num_nodes = data.num_nodes
        edge_index, edge_weight = get_laplacian(
            data.edge_index,
            data.edge_weight,
            normalization='sym',
            num_nodes=num_nodes,
        )

        L = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes)[0]
        #print("SHAPE", L.shape)

        eig_vals, eig_vecs = eig_fn(L, **self.kwargs)
        eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
        #data = add_node_attr(data, eig_vecs, attr_name=self.attr_name)

        if num_nodes < self.k + 2:
            padding = self.k + 2 - num_nodes
            eig_vecs = np.concatenate((eig_vecs, np.zeros((num_nodes, padding), dtype=np.float32)), axis=1)
        pe = torch.from_numpy(eig_vecs[:, 1:self.k + 1])
        sign = -1 + 2 * torch.randint(0, 2, (self.k, ))
        pe *= sign

        data = add_node_attr(data, pe, attr_name=self.attr_name)
        return data

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

    if args.dataset == 'RANDOM':
        dataset = RandomGraphDataset(root='datasets/random',
                    num_graphs=args.num_graphs,
                    num_nodes_per_graph=args.num_nodes_per_graph,
                    edge_probability=args.edge_probability,
                    pre_transform=pre_transform,
                    transform=transform)
    elif args.dataset == 'TU':
        dataset = TUDataset(root=f'datasets',
                    name=args.TUdataset_name,
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
