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
from model.parsing import read_params_from_folder

def construct_grad_layer(args):
    if args.problem_type == 'max_cut':
        return MaxCutGradLayer()
    elif args.problem_type == 'vertex_cover':
        return VertexCoverGradLayer()
    else:
        raise ValueError(f"construct_grad_layer got unsupported problem_type {problem_type}")

def construct_model(args):
    if args.model_type == 'LiftMP':
        model = LiftNetwork(
          grad_layer=construct_grad_layer(args),
          in_channels=args.rank,
          num_layers=args.num_layers,
        )
    elif args.model_type == 'FullMP':
        model = LiftProjectNetwork(
          grad_layer=construct_grad_layer(args),
          in_channels=args.rank,
          num_layers_lift=args.num_layers - args.num_layers_project,
          num_layers_project=args.num_layers_project,
        )
    elif args.problem_type == 'max_cut' and args.model_type == "ProjectMP":
        # must have lift network to train.
        assert args.lift_file is not None
        model = LiftProjectNetwork(
          grad_layer=construct_grad_layer(args),
          in_channels=args.rank,
          num_layers_lift=args.num_layers - args.num_layers_project,
          num_layers_project=args.num_layers_project,
          lift_file=args.lift_file,
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

    def forward(self, x, edge_index, **kwargs):
        edge_weight = kwargs['edge_weight']
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight[:, None]

    def update(self, aggr_out, x, edge_weight):
        return aggr_out

class VertexCoverGradLayer(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index, **kwargs):
        node_weight = kwargs['node_weight']
        vc_penalty = kwargs['vc_penalty']
        return self.propagate(edge_index, x=x, node_weight=node_weight, vc_penalty=vc_penalty)

    def message(self, x_i, x_j, node_weight, vc_penalty):
        e1 = torch.zeros_like(x_j) # (num_edges, hidden)
        e1[:, 0] = 1
        # take (e1 - x_i)[e] . (e1 - x_j)[e] for each row e
        phi = torch.sum((e1 - x_i) * (e1 - x_j), dim=1)[:, None] # (num_edges, 1)
        scaling = vc_penalty * phi
        return (-e1 + x_j) * scaling # (num_edges, hidden)

    def update(self, aggr_out, x, node_weight, vc_penalty):
        # aggr_out is the sum of gradients from constraints coming from neighbors
        e1 = torch.zeros_like(aggr_out) # (num_edges, hidden)
        e1[:, 0] = 1
        # add e1 for gradient of x_i and take negative for descent direction
        grad = -(aggr_out + 0.5 * e1 * node_weight.view(-1, 1))
        return grad

class LiftLayer(torch.nn.Module):
    def __init__(self, grad_layer, in_channels):
        super().__init__()
        self.grad_layer = grad_layer
        self.lin = Linear(2*in_channels, in_channels)

    def forward(self, x, edge_index, **kwargs):
        grads = self.grad_layer(x, edge_index, **kwargs)
        norm_grads = F.normalize(grads, dim=1)
        out = torch.cat((x, norm_grads), 1)
        out = self.lin(out)
        out = F.normalize(out, dim=1)
        return out

class LiftNetwork(torch.nn.Module):
    def __init__(self, grad_layer, in_channels, num_layers=12):
        super().__init__()
        self.layers = [LiftLayer(grad_layer, in_channels) for _ in range(num_layers)]
        for i, layer in enumerate(self.layers):
            self.add_module(f"layer_{i}", layer)

    def forward(self, x, edge_index, **kwargs):
        for l in self.layers:
            x = l(x, edge_index, **kwargs)
        return x

# Nearly identical to the lift layer. The big difference is that we no longer normalize in update.
class ProjectLayer(torch.nn.Module):
    def __init__(self, grad_layer, in_channels):
        super().__init__()
        self.grad_layer = grad_layer
        self.lin = Linear(2*in_channels, in_channels)

    def forward(self, x, edge_index, **kwargs):
        grads = self.grad_layer(x, edge_index, **kwargs)
        out = torch.cat((x, grads), 1)
        out = self.lin(out)
        out = F.tanh(out)
        return out

class ProjectNetwork(torch.nn.Module):
    def __init__(self, grad_layer, in_channels, num_layers=8):
        super().__init__()
        self.layers = [ProjectLayer(grad_layer, in_channels) for _ in range(num_layers)]
        for i, layer in enumerate(self.layers):
            self.add_module(f"layer_{i}", layer)

    def forward(self, x, edge_index, **kwargs):
        for l in self.layers:
            x = l(x, edge_index, **kwargs)
        return x

class LiftProjectNetwork(torch.nn.Module):
    def __init__(self, in_channels, num_layers_lift, num_layers_project, grad_layer, lift_file=None):
        super().__init__()

        if lift_file is not None:
            print(f"loading from {lift_file}")
            # TODO: maybe check the lift arguments for consistency

            # load lift file
            class DotDict(dict):
                def __getattr__(self, key):
                    if key in self:
                        return self[key]
                    else:
                        raise AttributeError(f"'DotDict' object has no attribute '{key}'")
            self.lift_args = DotDict(read_params_from_folder(os.path.dirname(lift_file)))
            self.lift_net, _ = construct_model(self.lift_args)
            if self.lift_args.rank != in_channels:
                raise ValueError(f"As of right now, lift_args rank ({self.lift_args.rank}) must \
                                 equal the project network rank ({in_channels})")
            self.lift_net = load_model(self.lift_net, lift_file)

            # freeze it
            for param in self.lift_net.parameters():
                param.requires_grad = False
            
        else:
            self.lift_net = LiftNetwork(grad_layer, in_channels, num_layers=num_layers_lift)
        self.project_net = ProjectNetwork(grad_layer, in_channels, num_layers=num_layers_project)

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

    def forward(self, x, edge_index, **kwargs):
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

    def forward(self, x, edge_index, **kwargs):
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

    def forward(self, x, edge_index, **kwargs):
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

    def forward(self, x, edge_index, **kwargs):
        out = self.net(x=x, edge_index=edge_index)
        out = F.normalize(out, dim=1)
        return out
