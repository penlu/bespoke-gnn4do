# Parse arguments

import math
import os
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from datetime import datetime

# NOTE(JXuKitty): not sure about this yet, still debating the ipynb route
def add_train_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """

    # General arguments
    parser.add_argument('--problem_type', type=str, default='max_cut',
        choices=['max_cut', 'vertex_cover', 'max_clique'],
        help='What problem are we doing?',
    )
    parser.add_argument('--seed', type=str, default=0,
                        help='Torch random seed to use to initialize networks')
    parser.add_argument('--prefix', type=str, default="",
                        help='Folder name prefix')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='RANDOM', choices=['RANDOM', 'TU'],
                        help='Dataset to use')

    # Arguments for random graphs
    parser.add_argument('--num_graphs', type=int, default=10000,
                        help='When using random graphs, how many to generate?')
    parser.add_argument('--num_nodes_per_graph', type=int, default=100,
                        help='When using random graphs, how many nodes per graph?')
    parser.add_argument('--edge_probability', type=float, default=0.15,
                        help='When using random dataset, what probability per edge in graph?')

    # Arguments for TU datasets
    parser.add_argument('--TUdataset_name', type=str, default='PROTEINS',
                        help='When using TU dataset, which dataset to use?')

    # Model construction arguments
    parser.add_argument('--model_type', type=str, default='LiftMP', choices=['LiftMP', 'FullMP', 'GIN', 'GAT', 'GCNN', 'GatedGCNN'],
                        help='Which type of model to use')
    parser.add_argument('--num_layers', type=int, default=12,
                        help='How many layers?')
    parser.add_argument('--num_layers_project', type=int, default=2,
                        help='How many projection layers? (when using FullMP)')
    parser.add_argument('--rank', type=int, default=2,
                        help='How many dimensions for the vectors at each node, i.e. what rank is the solution matrix?')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Model dropout')

    # Training parameters
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs', type=float, default=100,
                        help='Training epoch count')
    parser.add_argument('--valid_epochs', type=int, default=1,
                        help='Run validation every N epochs (0 to never run validation)')
    parser.add_argument('--save_epochs', type=int, default=0,
                        help='Save model every N epochs (0 to only save at end of training)')

    # TODO need some params for how often to run validation, what validation to run, how often to save

def modify_train_args(args: Namespace):
    """
    Modifies and validates training arguments in place.

    :param args: Arguments.
    """
    print("device", torch.cuda.is_available())
    setattr(
        args, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    # TODO add real logger functionality
    args.log_dir = "training_runs/" + args.prefix + '_' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

def parse_train_args() -> Namespace:
    """
    Parses arguments for training (includes modifying/validating arguments).

    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()
    modify_train_args(args)

    return args
