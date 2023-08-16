# load dataset
from torch_geometric.datasets import TUDataset

def construct_loader(args):
    dataset = TUDataset(root='/tmp/'+ args.TUdataset_name, name=args.TUdataset_name)
    # TODO