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
from utils.parsing import read_params_from_folder
from problem.problems import get_problem

def construct_model(args):
    if args.model_type == 'LiftMP':
        model = LiftNetwork(
          grad_layer=AutogradLayer(loss_fn=get_problem(args).loss),
          in_channels=args.rank,
          num_layers=args.num_layers,
          repeat_lift_layers=args.repeat_lift_layers,
        )
    elif args.model_type == 'FullMP':
        model = LiftProjectNetwork(
          grad_layer=AutogradLayer(loss_fn=get_problem(args).loss),
          in_channels=args.rank,
          num_layers_lift=args.num_layers - args.num_layers_project,
          num_layers_project=args.num_layers_project,
          repeat_lift_layers=args.repeat_lift_layers,
        )
    elif args.model_type == "ProjectMP":
        # must have lift network to train.
        assert args.lift_file is not None
        model = LiftProjectNetwork(
          grad_layer=AutogradLayer(loss_fn=get_problem(args).loss),
          in_channels=args.rank,
          num_layers_lift=args.num_layers - args.num_layers_project,
          num_layers_project=args.num_layers_project,
          lift_file=args.lift_file,
          repeat_lift_layers=args.repeat_lift_layers,
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

# use autograd on a given loss function to compute gradients
class AutogradLayer(torch.nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self._loss_fn = loss_fn

    def forward(self, x, batch):
        # calculate the lift loss and take the gradient w.r.t. the input x
        # the result is expected to be autodiffable, and training should be unaffected
        with torch.enable_grad():
            x.requires_grad_(True)
            loss = self._loss_fn(x, batch)
            grad = torch.autograd.grad(loss, x, create_graph=True)[0]
            return grad

class LiftLayer(torch.nn.Module):
    def __init__(self, grad_layer, in_channels):
        super().__init__()
        self.grad_layer = grad_layer
        self.lin = Linear(2*in_channels, in_channels)

    def forward(self, x, batch):
        grads = self.grad_layer(x, batch)
        norm_grads = F.normalize(grads, dim=1)
        out = torch.cat((x, norm_grads), 1)
        out = self.lin(out)
        out = F.normalize(out, dim=1)
        return out

class LiftNetwork(torch.nn.Module):
    def __init__(self, grad_layer, in_channels, num_layers=12, repeat_lift_layers=None):
        super().__init__()
        if repeat_lift_layers is not None:
            # the number of layers must equal the length of the repeat array.
            assert(len(repeat_lift_layers) == num_layers)
        self.layers = [LiftLayer(grad_layer, in_channels) for _ in range(num_layers)]
        for i, layer in enumerate(self.layers):
            self.add_module(f"layer_{i}", layer)
        
        if repeat_lift_layers is None:
            repeat_lift_layers = [1 for _ in range(num_layers)]
        self.repeat_lift_layers = repeat_lift_layers

    def forward(self, x, batch):
        for l, repeat_l in zip(self.layers, self.repeat_lift_layers):
            for _ in range(repeat_l):
                x = l(x, batch)
        return x

# Nearly identical to the lift layer. The big difference is that we no longer normalize in update.
class ProjectLayer(torch.nn.Module):
    def __init__(self, grad_layer, in_channels):
        super().__init__()
        self.grad_layer = grad_layer
        self.lin = Linear(2*in_channels, in_channels)

    def forward(self, x, batch):
        grads = self.grad_layer(x, batch)
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

    def forward(self, x, batch):
        for l in self.layers:
            x = l(x, batch)
        return x

class LiftProjectNetwork(torch.nn.Module):
    def __init__(self, in_channels, num_layers_lift, num_layers_project, grad_layer, lift_file=None, repeat_lift_layers=None):
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
            self.lift_net = LiftNetwork(grad_layer, in_channels, num_layers=num_layers_lift, repeat_lift_layers=repeat_lift_layers)
        self.project_net = ProjectNetwork(grad_layer, in_channels, num_layers=num_layers_project)

    def forward(self, x, batch):
        out = self.lift_net(x, batch)
        # TODO randomly rotate here
        out = self.project_net(out, batch)
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

    def forward(self, x, batch):
        out = self.net(x=x, edge_index=batch.edge_index)
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

    def forward(self, x, batch):
        out = self.net(x=x, edge_index=batch.edge_index)
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

    def forward(self, x, batch):
        out = self.net(x=x, edge_index=batch.edge_index)
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

    def forward(self, x, batch):
        out = self.net(x=x, edge_index=batch.edge_index)
        out = F.normalize(out, dim=1)
        return out

class LiftLayerv2(torch.nn.Module):
    def __init__(self, grad_layer, in_channels):
        super().__init__()
        self.grad_layer = grad_layer
        self.lin = Linear(2*in_channels, in_channels)

    def forward(self, x, batch):
        grads = self.grad_layer(x, batch)
        out = torch.cat((x, grads), 1)
        out = self.lin(out)
        out = F.normalize(out, dim=1)
        return out

class ProjectNetwork_r1(torch.nn.Module):
    def __init__(self, grad_layer, in_channels, num_layers=6):
        super().__init__()
        self.layers = [LiftLayerv2(grad_layer, in_channels) for _ in range(num_layers)]
        for i, layer in enumerate(self.layers):
            self.add_module(f"layer_{i}", layer)
        self.lin = Linear(in_channels, 1)

    def forward(self, x, batch):
        for l in self.layers:
            x = F.leaky_relu(l(x, batch)+x,0.01)
        x = F.normalize(x)
        return x
