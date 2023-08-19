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
    if args.problem_type == 'max_cut' and args.model_type == 'LiftMP':
        model = MaxCutLiftNetwork(
          in_channels=args.rank,
          num_layers=args.num_layers,
        )
    elif args.problem_type == 'max_cut' and args.model_type == 'FullMP':
        model = MaxCutLiftProjectNetwork(
          in_channels=args.rank,
          num_layers_lift=args.num_layers - args.num_layers_project,
          num_layers_project=args.num_layers_project,
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

# This network concatenates the neighborhood gradient to each vector in its input.
class MaxCutGradLayer(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index, edge_weights):
        return self.propagate(edge_index, x=x, edge_weights=edge_weights)

    def message(self, x_j, edge_weights):
        return x_j * edge_weights[:, None]

    def update(self, aggr_out, x, edge_weights):
        return aggr_out

class MaxCutLiftLayer(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.grad_layer = MaxCutGradLayer()
        self.lin = Linear(2*in_channels, in_channels)

    def forward(self, x, edge_index, edge_weights):
        grads = self.grad_layer(x, edge_index, edge_weights)
        norm_grads = F.normalize(grads, dim=1)
        out = torch.cat((x, norm_grads), 1)
        out = self.lin(out)
        out = F.normalize(out, dim=1)
        return out

class MaxCutLiftNetwork(torch.nn.Module):
    def __init__(self, in_channels, num_layers=12):
        super().__init__()
        self.layers = [MaxCutLiftLayer(in_channels) for _ in range(num_layers)]
        for i, layer in enumerate(self.layers):
            self.add_module(f"layer_{i}", layer)

    def forward(self, x, edge_index, edge_weights):
        for l in self.layers:
            x = l(x, edge_index, edge_weights)
        return x

# Nearly identical to the lift layer. The big difference is that we no longer normalize in update.
class MaxCutProjectLayer(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.grad_layer = MaxCutGradLayer()
        self.lin = Linear(2*in_channels, in_channels)

    def forward(self, x, edge_index, edge_weights):
        grads = self.grad_layer(x, edge_index, edge_weights)
        out = torch.cat((x, grads), 1)
        out = self.lin(out)
        out = F.tanh(out)
        return out

class MaxCutProjectNetwork(torch.nn.Module):
    def __init__(self, in_channels, num_layers=8):
        super().__init__()
        self.layers = [MaxCutProjectLayer(in_channels) for _ in range(num_layers)]
        for i, layer in enumerate(self.layers):
            self.add_module(f"layer_{i}", layer)

    def forward(self, x, edge_index, edge_weights):
        for l in self.layers:
            x = l(x, edge_index, edge_weights)
        return x

class MaxCutLiftProjectNetwork(torch.nn.Module):
    def __init__(self, in_channels, num_layers_lift, num_layers_project):
        super().__init__()
        self.lift_net = MaxCutLiftNetwork(in_channels, num_layers=num_layers_lift)
        self.project_net = MaxCutProjectNetwork(in_channels, num_layers=num_layers_project)

    def forward(self, x, edge_index, edge_weights):
        out = self.lift_net(x, edge_index, edge_weights)
        # TODO randomly rotate here
        out = self.project_net(out, edge_index, edge_weights)
        return out

# TODO graph isomorphism network

# TODO graph attention network

# TODO graph convnet

# TODO gated graph convnet
