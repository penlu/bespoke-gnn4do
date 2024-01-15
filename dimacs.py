# Generate DIMACS files

import itertools
import json
import os
import time
import traceback

import torch
from torch.utils.data import IterableDataset, random_split
from torch_geometric.utils.convert import to_networkx

from data.loader import construct_dataset
from data.sat import dimacs_printer
from problem.problems import get_problem
from problem.baselines import e1_projector, random_hyperplane_projector
from utils.parsing import parse_baseline_args
from utils.graph_utils import complement_graph

if __name__ == '__main__':
    # parse args
    args = parse_baseline_args()
    print(args)
    torch.manual_seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)

    # save params
    args.device = str(args.device)
    json.dump(vars(args), open(os.path.join(args.log_dir, 'params.txt'), 'w'))
    outfile = open(os.path.join(args.log_dir, 'results.jsonl'), 'w')

    # get data
    dataset = construct_dataset(args)
    if isinstance(dataset, IterableDataset):
        val_set = list(itertools.islice(dataset, 1000))
        test_set = list(itertools.islice(dataset, 1000))
    else:
        train_size = int(args.train_fraction * len(dataset))
        val_size = (len(dataset) - train_size)//2
        test_size = len(dataset) - train_size - val_size

        generator = torch.Generator().manual_seed(args.split_seed)
        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    for i, data in enumerate(test_set):
        dimacs = dimacs_printer(data.num_vars, data.num_clauses, data.clause_index, data.signs)
        with open(os.path.join(args.log_dir, f'dimacs_{i}_{data.num_vars}_{data.num_clauses}.txt'), 'w') as f:
            f.write(dimacs)
