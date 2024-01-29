# Loss functions and their associated gradients

import numpy as np
import time

import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_torch_csr_tensor, to_torch_coo_tensor
from functools import partial

# X should have shape (N, r)
def max_cut_obj(X, batch):
    # attach edge weights if they're not already present
    if not hasattr(batch, 'edge_weight') or batch.edge_weight is None:
        num_edges = batch.edge_index.shape[1]
        batch.edge_weight = torch.ones(num_edges, device=X.device)

    # compute loss
    X0 = X[batch.edge_index[0]]
    X1 = X[batch.edge_index[1]]
    edges = torch.sum(X0 * X1, dim=1)
    obj = torch.sum(edges * batch.edge_weight)
    return obj

def vertex_cover_obj(X, batch):
    # attach node weights if they're not already present
    N = batch.num_nodes
    if not hasattr(batch, 'node_weight') or batch.node_weight is None:
        batch.node_weight = torch.ones(N, device=X.device)

    # lift adopts e1 = (1,0,...,0) as 1
    # count number of vertices: \sum_{i \in [N]} w_i(1+x_i)/2
    obj = torch.inner(torch.ones(N).to(X.device) + X[:, 0], batch.node_weight) / 2.
    return obj

def vertex_cover_constraint(X, batch):
    N = batch.num_nodes

    # now calculate penalty for uncovered edges
    # phi is matrix of dimension N by N for error per edge
    # phi_ij = 1 - <x_i + x_j,e_1> + <x_i,x_j> for (i,j) \in Edges
    # phi_ij = <x_i - e1, x_j - e1> for (i, j) \in Edges
    e1 = torch.zeros_like(X) # (num_edges, hidden)
    e1[:, 0] = 1
    Xm = X - e1

    penalties = torch.sum(Xm[batch.edge_index[0]] * Xm[batch.edge_index[1]], dim=1) / 2.
    constraint = torch.sum(penalties * penalties)
    return constraint

# we are receiving the _complement_ of the target graph
# TODO fix this
def max_clique_loss(X, batch, penalty=2):
    return vertex_cover_loss(X, batch, penalty=penalty)

def max_cut_score(args, X, example):
    # convert numpy array to torch tensor
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X)
    if len(X.shape) == 1:
        X = X[:, None]
    N = example.num_nodes
    E = example.edge_index.shape[1]
    return (E - max_cut_obj(X, example)) / 2.

def vertex_cover_score(args, X, example):
    # convert numpy array to torch tensor
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X)
    if len(X.shape) == 1:
        X = X[:, None]
    return - (vertex_cover_obj(X, example) + vertex_cover_constraint(X, example))

# we are receiving the _complement_ of the target graph
# the score is N - k where k is the vertex cover size
def max_clique_score(args, X, example):
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X)
    if len(X.shape) == 1:
        X = X[:, None]
    N = example.num_nodes
    return N - (vertex_cover_obj(X, example) + vertex_cover_constraint(X, batch))
