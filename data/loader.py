# Dataset loading functionality

import torch
import networkx
import numpy
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import TUDataset
from torch_geometric.utils.convert import from_networkx
from utils.graph_utils import gen_graph

# TODO check if this works right
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
        random_state = numpy.random.RandomState(seed)
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

def construct_loader(args):
    if args.dataset == 'RANDOM':
        dataset = RandomGraphDataset(root='/tmp/random',
                    num_graphs=args.num_graphs,
                    num_nodes_per_graph=args.num_nodes_per_graph,
                    edge_probability=args.edge_probability)
    elif args.dataset == 'TU':
        dataset = TUDataset(root=f'/tmp/{args.TUdataset_name}', name=args.TUdataset_name)
    else:
        raise ValueError(f"Unimplemented dataset {args.dataset}. Expected RANDOM or TU.")

    return dataset
