# Models
import matplotlib.pyplot as plt

import numpy as np
import time

import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter, Sequential
from torch.optim import Adam

from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models import GAT
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.nn.models.basic_gnn import BasicGNN

def construct_model(args):
    if args.model_type == 'MP':
        model = SimpleLiftNetwork(
          in_channels=args.rank,
          num_layers=args.num_layers,
          problem_type=args.problem_type,
        )
    elif args.model_type == 'GIN':
        raise NotImplementedError('GIN not yet implemented')
    elif args.model_type == 'GAT':
        raise NotImplementedError('GAT not yet implemented')
    elif args.model_type == 'GCNN':
        raise NotImplementedError('GCNN not yet implemented')
    elif args.model_type == 'GatedGCNN':
        raise NotImplementedError('GatedGCNN not yet implemented')
    else:
        raise ValueError(f'Got unexpected model_type {args.model_type}')

    opt = Adam(model.parameters(), args.lr)

    return model, opt

# TODO grad functions: right now this is only appropriate for max cut
class SimpleLiftLayer(MessagePassing):
    def __init__(self, in_channels, problem_type):
        super().__init__(aggr='add')
        self.problem_type = problem_type
        # TODO get gradient computation layer here
        self.lin2 = Linear(2*in_channels, in_channels)

    def forward(self, x, edge_index, edge_weights):
        out = self.propagate(edge_index, x=x, edge_weights=edge_weights)
        return out

    # XXX message is gradients?
    def message(self, x_j, edge_weights):
        x_j = x_j * edge_weights[:,None]
        return x_j
    
    def update(self, aggr_out, x):
        norm_aggr = F.normalize(aggr_out,dim=1)
        concat = torch.cat((norm_aggr, x), 1)
        out = self.lin2(concat)
        out = F.normalize(out, dim=1)
        return out

class SimpleLiftNetwork(torch.nn.Module):
    def __init__(self, in_channels, num_layers=12, problem_type='max_cut'):
        super().__init__()
        self.layers = [SimpleLiftLayer(in_channels, problem_type) for _ in range(num_layers)]
        for i, layer in enumerate(self.layers):
            self.add_module(f"layer_{i}", layer)

    def forward(self, x, edge_index, edge_weights):
        for l in self.layers:
            x = l(x, edge_index, edge_weights)
        return x

# Nearly identical to the lift layer. The big difference is that we no longer normalize in update.
class SimpleProjectLayer(MessagePassing):
    def __init__(self, in_channels, problem_type):
        super().__init__(aggr='add')
        self.problem_type = problem_type
        # TODO get gradient computation layer here
        self.lin2 = Linear(2*in_channels, in_channels)

    def forward(self, x, edge_index, edge_weights):
        out = self.propagate(edge_index, x=x, edge_weights=edge_weights)
        return out

    # XXX message is gradients?
    def message(self, x_j, edge_weights):
        # TODO grad functions: right now this is only appropriate for max cut
        x_j = x_j * edge_weights[:,None]
        return x_j
    
    def update(self, aggr_out, x):
        concat = torch.cat((norm_aggr, x), 1)
        out = self.lin2(concat)
        out = torch.tanh(out)
        return out

class SimpleProjectNetwork(torch.nn.Module):
    def __init__(self, in_channels, num_layers=8, problem_type='max_cut'):
        super().__init__()
        self.layers = [SimpleProjectLayer(in_channels, problem_type) for _ in range(num_layers)]
        for i, layer in enumerate(self.layers):
            self.add_module(f"layer_{i}", layer)

    def forward(self, x, edge_index, edge_weights):
        for l in self.layers:
            x = l(x, edge_index, edge_weights)
        return x

# TODO graph isomorphism network

# TODO graph attention network

# TODO graph convnet

# TODO gated graph convnet
