# Models
import matplotlib.pyplot as plt

import numpy as np
import time
import os

import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter, Sequential
from torch.optim import Adam

from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models import GAT, GIN, GCN
from torch_geometric.nn.conv import GatedGraphConv

from model.more_models import NegationGAT
from model.saving import load_model

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
        model = GINLiftNetwork(args)
    elif args.model_type == 'GAT':
        model = GATLiftNetwork(args)
    elif args.model_type == 'GCNN':
        model = GCNLiftNetwork(args)
    elif args.model_type == 'GatedGCNN':
        model = GatedGCNLiftNetwork(args)
    elif args.model_type == 'NegationGAT':
        model = NegationGAT(in_channels=args.rank, 
                            hidden_channels=args.hidden_channels, 
                            dropout=args.dropout, 
                            v2=True, norm=args.norm, 
                            num_layers=args.num_layers)
    else:
        raise ValueError(f'Got unexpected model_type {args.model_type}')

    if args.finetune_from is not None:
        # load in model for finetuning
        model = load_model(model, args.finetune_from)
        model.to(args.device)

    opt = Adam(model.parameters(), args.lr)

    return model, opt

# This network concatenates the neighborhood gradient to each vector in its input.
class MaxCutGradLayer(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight[:, None]

    def update(self, aggr_out, x, edge_weight):
        return aggr_out

class MaxCutLiftLayer(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.grad_layer = MaxCutGradLayer()
        self.lin = Linear(2*in_channels, in_channels)

    def forward(self, x, edge_index, edge_weight):
        grads = self.grad_layer(x, edge_index, edge_weight)
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

    def forward(self, x, edge_index, edge_weight):
        for l in self.layers:
            x = l(x, edge_index, edge_weight)
        return x

# Nearly identical to the lift layer. The big difference is that we no longer normalize in update.
class MaxCutProjectLayer(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.grad_layer = MaxCutGradLayer()
        self.lin = Linear(2*in_channels, in_channels)

    def forward(self, x, edge_index, edge_weight):
        grads = self.grad_layer(x, edge_index, edge_weight)
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

    def forward(self, x, edge_index, edge_weight):
        for l in self.layers:
            x = l(x, edge_index, edge_weight)
        return x

class MaxCutLiftProjectNetwork(torch.nn.Module):
    def __init__(self, in_channels, num_layers_lift, num_layers_project):
        super().__init__()
        self.lift_net = MaxCutLiftNetwork(in_channels, num_layers=num_layers_lift)
        self.project_net = MaxCutProjectNetwork(in_channels, num_layers=num_layers_project)

    def forward(self, x, edge_index, edge_weight):
        out = self.lift_net(x, edge_index, edge_weight)
        # TODO randomly rotate here
        out = self.project_net(out, edge_index, edge_weight)
        return out

# graph isomorphism network
# TODO version with gradients, version allowing negation of neighbors
class GINLiftNetwork(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = GIN(in_channels=args.rank,
            hidden_channels=args.hidden_channels,
            dropout=args.dropout,
            norm=args.norm,
            num_layers=args.num_layers)

    def forward(self, x, edge_index, edge_weight):
        out = self.net(x=x, edge_index=edge_index)
        out = F.normalize(out, dim=1)
        return out

# graph attention network
# TODO version with gradients, version allowing negation of neighbors
class GATLiftNetwork(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = GAT(in_channels=args.rank,
            hidden_channels=args.hidden_channels,
            dropout=args.dropout,
            v2=True,
            norm=args.norm,
            num_layers=args.num_layers,
            heads=args.heads)

    def forward(self, x, edge_index, edge_weight):
        out = self.net(x=x, edge_index=edge_index)
        out = F.normalize(out, dim=1)
        return out

# graph convnet
# TODO version with gradients, version allowing negation of neighbors
class GCNLiftNetwork(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = GCN(in_channels=args.rank,
            hidden_channels=args.hidden_channels,
            dropout=args.dropout,
            norm=args.norm,
            num_layers=args.num_layers)

    def forward(self, x, edge_index, edge_weight):
        out = self.net(x=x, edge_index=edge_index)
        out = F.normalize(out, dim=1)
        return out

# gated graph convnet
# TODO version with gradients, version allowing negation of neighbors
class GatedGCNLiftNetwork(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        # TODO - does this need more params? is out_channel correct?
        self.net = GatedGraphConv(out_channels=args.hidden_channels,
            num_layers=args.num_layers)

    def forward(self, x, edge_index, edge_weight):
        out = self.net(x=x, edge_index=edge_index)
        out = F.normalize(out, dim=1)
        return out
