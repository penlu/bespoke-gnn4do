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
    A = to_torch_csr_tensor(edge_index, size=X.shape[0])
    XX = torch.matmul(X, torch.transpose(X, -1, -2))
    obj = torch.matmul(A, XX).diagonal(dim1=-1, dim2=-2).sum(-1) / 2.

    return obj

def get_score_fn(args):
    if args.problem_type == 'max_cut':
        return max_cut_score
    elif args.problem_type == 'vertex_cover':
        return vertex_cover_score
    elif args.problem_type == 'max_clique':
        raise NotImplementedError('max_clique loss not yet implemented')

def max_cut_score(args, X, example):
    # convert numpy array to torch tensor
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X)
    if len(X.shape) == 1:
        X = X[:, None]
    N = example.num_nodes
    edge_index = example.edge_index.to(X.device)
    A = to_torch_csr_tensor(edge_index, size=N)
    E = edge_index.shape[1]
    XX = torch.matmul(X, torch.transpose(X, -1, -2))
    obj = torch.matmul(A, XX).diagonal(dim1=-1, dim2=-2).sum(-1) / 2.

    return (E - obj) / 2., 0.

def vertex_cover_score():
    raise NotImplementedError()
