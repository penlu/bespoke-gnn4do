# Parse arguments

import math
import os
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from datetime import datetime
import json
import hashlib


def add_general_args(parser: ArgumentParser):
    # General arguments
    parser.add_argument('--problem_type', type=str, default='max_cut',
        choices=['max_cut', 'vertex_cover', 'max_clique'],
        help='What problem are we doing?',
    )
    parser.add_argument('--seed', type=str, default=0,
                        help='Torch random seed to use to initialize networks')
    parser.add_argument('--prefix', type=str,
                        help='Folder name for run outputs if desired (will default to run timestamp)')

def add_dataset_args(parser: ArgumentParser):
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='RANDOM', choices=['RANDOM', 'TU'],
                        help='Dataset type to use')

    # Arguments for random graphs
    parser.add_argument('--num_graphs', type=int, default=1000,
                        help='When using random graphs, how many to generate?')
    parser.add_argument('--num_nodes_per_graph', type=int, default=100,
                        help='When using random graphs, how many nodes per graph?')
    parser.add_argument('--edge_probability', type=float, default=0.15,
                        help='When using random graphs, what probability per edge in graph?')

    # Arguments for TU datasets
    parser.add_argument('--TUdataset_name', type=str, default=None,
                        help='When using TU dataset, which dataset to use?')

    # Positional encoding arguments
    parser.add_argument('--positional_encoding', type=str, default=None,
                        choices=['laplacian_eigenvector', 'random_walk'],
                        help='Use a positional encoding?')
    parser.add_argument('--pe_dimension', type=int, default=8,
                        help='Dimensionality of the positional encoding')

def add_train_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    # Model construction arguments
    parser.add_argument('--model_type', type=str, default='LiftMP', choices=['LiftMP', 'FullMP', 'GIN', 'GAT', 'GCNN', 'GatedGCNN', 'NegationGAT', 'ProjectMP'],
                        help='Which type of model to use')
    parser.add_argument('--num_layers', type=int, default=12,
                        help='How many layers?')
    parser.add_argument('--num_layers_project', type=int, default=4,
                        help='How many projection layers? (when using FullMP)')
    parser.add_argument('--rank', type=int, default=32,
                        help='How many dimensions for the vectors at each node, i.e. what rank is the solution matrix?')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Model dropout')
    parser.add_argument('--hidden_channels', type=int, default=32,
                        help='Dimensions of the hidden channels')
    parser.add_argument('--norm', type=str, default="BatchNorm",
                        help='Normalization to use')
    parser.add_argument('--heads', type=int, default=4,
                        help='number of heads for GAT')
    parser.add_argument('--finetune_from', type=str, default=None, 
                        help="model file to load weights from for finetuning")
    parser.add_argument('--lift_file', type=str, default=None, 
                        help="model file from which to load lift network")

    # Training parameters
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--valid_freq', type=int, default=0,
                        help='Run validation every N steps/epochs (0 to never run validation)')
    parser.add_argument('--save_freq', type=int, default=0,
                        help='Save model every N steps/epochs (0 to only save at end of training)')
    parser.add_argument('--vc_penalty', type=float, default=None,
                        help='Penalty for failed cover in vertex cover')

    parser.add_argument('--stepwise', type=bool, default=True,
                        help='Train by number of gradient steps or number of epochs?')
    parser.add_argument('--steps', type=int, default=50000,
                        help='Training step count')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epoch count')

    # TODO need some params for how often to run validation, what validation to run, how often to save
    parser.add_argument('--valid_fraction', type=float, default=0.2,
                        help='Fraction of data to set aside for validation. XXX currently not used')

def hash_dict(d):
    # Convert the dictionary to a sorted tuple of key-value pairs
    sorted_items = str(tuple(sorted(d.items())))
    #print(sorted_items)
    
    # Hash the tuple
    hash_value = hashlib.sha256(sorted_items.encode()).hexdigest()
    return hash_value

def modify_train_args(args: Namespace):
    """
    Modifies and validates training arguments in place.

    :param args: Arguments.
    """  

    if args.finetune_from is not None:
        model_folder = os.path.dirname(args.finetune_from)
        model_args = read_params_from_folder(model_folder)
        print(model_args.keys())
        relevant_keys = ['problem_type', 
                         'model_type', 
                         'num_layers', 
                         'num_layers_project', 
                         'rank', 'dropout', 
                         'hidden_channels', 
                         'norm', 
                         'heads', 
                         'positional_encoding',
                         'pe_dimension']
        for k in relevant_keys:
            setattr(args, k, model_args[k])

    # TODO add real logger functionality
    if args.prefix is None:
        args.log_dir = "training_runs/" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    else:
        hashed_params = hash_dict(vars(args))
        #print(hashed_params)
        args.log_dir = "training_runs/" + args.prefix + f"_paramhash:{hashed_params}"
    print("device", torch.cuda.is_available())
    setattr(
        args, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

def parse_train_args() -> Namespace:
    """
    Parses arguments for training (includes modifying/validating arguments).

    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    add_general_args(parser)
    add_train_args(parser)
    add_dataset_args(parser)
    args = parser.parse_args()
    modify_train_args(args)

    # TODO: checks in a separate function?
    if args.vc_penalty is not None and args.problem_type == 'max_cut':
        raise ValueError(f"vc_penalty set for max cut")

    return args

def read_params_from_folder(model_folder):
    with open(os.path.join(model_folder, 'params.txt'), 'r') as f:
        model_args = json.load(f)
    
    return model_args

def parse_test_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--model_folder', type=str, default=None,
                        help='folder to look in.')
    parser.add_argument('--model_file', type=str, default=None,
                        help='model file')
    add_general_args(parser)
    add_dataset_args(parser)
    args = parser.parse_args()

    # read params from model folder.
    model_args = read_params_from_folder(args.model_folder)

    # set device
    model_args[ "device"] =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get relevant keys
    argkeys = vars(args).keys()
    for k, v in model_args.items():
        if k not in argkeys:
            setattr(args, k, v)

    return args

def modify_baseline_args(args: Namespace):
    """
    Modifies and validates training arguments in place.

    :param args: Arguments.
    """
    print("Is GPU available?", torch.cuda.is_available())
    setattr(
        args, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    # TODO add real logger functionality
    # TODO: decide what to name the log dir.
    if args.prefix is None:
        args.log_dir = "baseline_runs/" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    else:
        hashed_params = hash_dict(vars(args))
        #print(hashed_params)
        args.log_dir = "baseline_runs/" + args.prefix + f"_paramhash:{hashed_params}"
    args.batch_size = 1

def parse_baseline_args() -> Namespace:
    parser = ArgumentParser()
    add_general_args(parser)
    add_dataset_args(parser)
    parser.add_argument('--rank', type=int, default=32,
                        help='How many dimensions for the vectors at each node, i.e. what rank is the solution matrix?')
    parser.add_argument('--gurobi', type=bool, default=False,
                        help='Run Gurobi')
    parser.add_argument('--gurobi_timeout', type=float, default=5,
                        help='Timeout for Gurobi if desired')

    # TODO argument to control list of baselines to run?

    args = parser.parse_args()
    modify_baseline_args(args)
    return args
