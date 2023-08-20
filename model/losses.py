# Loss functions and their associated gradients

import numpy as np
import time

import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_torch_csr_tensor

def get_loss_fn(args):
    if args.problem_type == 'max_cut':
        return max_cut_loss
    elif args.problem_type == 'vertex_cover':
        raise NotImplementedError('vertex_cover loss not yet implemented')
    elif args.problem_type == 'max_clique':
        raise NotImplementedError('max_clique loss not yet implemented')

# X should have shape (N, r)
def max_cut_loss(X, edge_index):
    # compute loss
    A = to_torch_csr_tensor(edge_index)
    XX = torch.matmul(X, torch.transpose(X, 0, 1))
    obj = torch.trace(torch.matmul(A, XX)) / 2.

    return obj
