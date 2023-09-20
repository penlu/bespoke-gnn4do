import json
import os
import sys

import torch

from data.loader import construct_dataset

if len(sys.argv) == 2:
    problem_type = sys.argv[1]
else:
    problem_type = 'max_cut'
print(problem_type)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# construct random
construct_dataset(dotdict({
    'dataset': 'RANDOM',
    'num_graphs': 1000,
    'num_nodes_per_graph': 100,
    'edge_probability': 0.15,
    'problem_type': problem_type,
}))

# construct ForcedRB
construct_dataset(dotdict({
    'dataset': 'ForcedRB',
    'num_graphs': 1000,
    'problem_type': problem_type,
}))

# construct many TU datasets
for TU_name in ['ENZYMES', 'PROTEINS', 'IMDB-BINARY', 'MUTAG', 'COLLAB']:
    construct_dataset(dotdict({
        'dataset': 'TU',
        'TUdataset_name': TU_name,
        'problem_type': problem_type,
    }))
