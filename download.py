import json
import os

import torch

from data.loader import construct_dataset

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
    'edge_probability': 0.15
}))

# construct ForcedRB
construct_dataset(dotdict({
    'dataset': 'ForcedRB',
    'num_graphs': 1000,
}))

# construct many TU datasets
for TU_name in ['ENZYMES', 'PROTEINS', 'IMDB-BINARY', 'MUTAG', 'COLLAB']:
    construct_dataset(dotdict({'dataset': 'TU', 'TUdataset_name': TU_name}))
