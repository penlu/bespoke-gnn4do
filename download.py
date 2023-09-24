import json
import os
import sys

import torch

from data.loader import construct_dataset, TU_datasets

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

# construct many TU datasets
for TU_name in TU_datasets:
    construct_dataset(dotdict({
        'dataset': TU_name,
        'problem_type': problem_type,
    }))
