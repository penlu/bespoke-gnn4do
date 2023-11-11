# Parse arguments

import math
import os
import argparse
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from datetime import datetime
import json
import hashlib

# argparse custom action for enforcing variable number of args for an option
# taken from https://stackoverflow.com/a/4195302
def required_length(nmin, nmax):
    class RequiredLength(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if not nmin <= len(values) <= nmax:
                msg='argument "{f}" requires between {nmin} and {nmax} arguments'.format(
                    f=self.dest, nmin=nmin, nmax=nmax)
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, values)
    return RequiredLength

def add_general_args(parser: ArgumentParser):
    # General arguments
    parser.add_argument('--problem_type', type=str, default='max_cut',
        choices=['max_cut', 'vertex_cover', 'max_clique', 'sat'],
        help='What problem are we doing?',
    )
    parser.add_argument('--seed', type=str, default=0,
                        help='Torch random seed to use to initialize networks')
    parser.add_argument('--prefix', type=str,
                        help='Folder name for run outputs if desired (will default to run timestamp)')

def add_dataset_args(parser: ArgumentParser):
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='ErdosRenyi',
                        choices=[
                            'ErdosRenyi',
                            'BarabasiAlbert',
                            'PowerlawCluster',
                            'WattsStrogatz',
                            'ForcedRB',
                            'ENZYMES',
                            'PROTEINS',
                            'IMDB-BINARY',
                            'MUTAG',
                            'COLLAB',
                            'REDDIT-MULTI-5K',
                            'REDDIT-MULTI-12K',
                            'REDDIT-BINARY',
                        ],
                        help='Dataset type to use')

    # Arguments for generated datasets
    parser.add_argument('--data_seed', type=int, default=0,
                        help='Seed to use for generated datasets (RANDOM and ForcedRB)')
    parser.add_argument('--parallel', type=int, default=0,
                        help='How many parallel workers to use for generating data?')
    parser.add_argument('--num_graphs', type=int, default=1000,
                        help='When using generated datasets, how many graphs to generate? (Ignored when using --infinite)')
    parser.add_argument('--infinite', type=bool, default=False,
                        help='When using generated datasets, do infinite generation?')

    # Some generated dataset parameters
    parser.add_argument('--gen_n', nargs='+', type=int, default=100, action=required_length(1, 2),
                        help='Range for the n parameter of generated dataset (usually number of vertices)')
    parser.add_argument('--gen_m', type=int, default=4,
                        help='m parameter of generated dataset (meaning varies)')
    parser.add_argument('--gen_k', type=int, default=4,
                        help='k parameter of generated dataset (meaning varies)')
    parser.add_argument('--gen_p', type=float, default=None,
                        help='p parameter of generated dataset (meaning varies)')

    # Arguments for ForcedRB graphs
    parser.add_argument('--RB_n', nargs=2, type=int, default=[10, 26],
                        help='For ForcedRB, how many disjoint cliques? This upper bounds maximum independent set size. Provide two numbers for range [a, b).')
    parser.add_argument('--RB_k', nargs=2, type=int, default=[5, 21],
                        help='For ForcedRB, how many nodes in each disjoint clique? Provide two numbers for range [a, b).')

    # Arguments for chordal graphs
    # TODO

    # Positional encoding arguments
    parser.add_argument('--positional_encoding', type=str, default=None,
                        choices=['laplacian_eigenvector', 'random_walk'],
                        help='Use a positional encoding?')
    parser.add_argument('--pe_dimension', type=int, default=8,
                        help='Dimensionality of the positional encoding')

    # Arguments for data loader
    parser.add_argument('--split_seed', type=int, default=0,
                        help='Seed to use for train/val/test split')

def add_train_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    # Model construction arguments
    parser.add_argument('--model_type', type=str, default='LiftMP', choices=['LiftMP', 'FullMP', 'GIN', 'GAT', 'GCNN', 'GatedGCNN', 'NegationGAT', 'ProjectMP', 'Nikos'],
                        help='Which type of model to use')
    parser.add_argument('--num_layers', type=int, default=12,
                        help='How many layers?')
    parser.add_argument('--repeat_lift_layers', nargs='+', type=int, default=None,
                        help='A list of the number of times each layer should be repeated')
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
    parser.add_argument('--valid_freq', type=int, default=1000,
                        help='Run validation every N steps/epochs (0 to never run validation)')
    parser.add_argument('--save_freq', type=int, default=1000,
                        help='Save model every N steps/epochs (0 to only save at end of training)')
    parser.add_argument('--penalty', type=float, default=1.,
                        help='Penalty for constraint violations')

    parser.add_argument('--stepwise', type=bool, default=True,
                        help='Train by number of gradient steps or number of epochs?')
    parser.add_argument('--steps', type=int, default=50000,
                        help='Training step count')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epoch count')

    parser.add_argument('--train_fraction', type=float, default=0.8,
                        help='Fraction of data to retain for training. Remainder goes to validation/testing.')

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
        print("WARNING, please check the list of relevant_keys")
        pretrain_config_keys = ['problem_type', 
                         'model_type', 
                         'num_layers', 
                         'num_layers_project', 
                         'rank', 'dropout', 
                         'hidden_channels', 
                         'norm', 
                         'heads', 
                         'positional_encoding',
                         'pe_dimension',
                         'repeat_lift_layers',
                         'lift_file']
        for k in pretrain_config_keys:
            setattr(args, k, model_args[k])

    # TODO add real logger functionality
    if args.prefix is None:
        args.log_dir = "training_runs/" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    else:
        hashed_params = hash_dict(vars(args))
        #print(hashed_params)
        args.log_dir = f"training_runs/{args.prefix}/paramhash:{hashed_params}"
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
    parser.add_argument('--test_prefix', type=str, default=None,
                        help='test output filename prefix')
    parser.add_argument('--use_val_set', type=bool, default=False,
                        help='use the validation set instead of the test set')
    #add_general_args(parser)
    #add_dataset_args(parser)
    args = parser.parse_args()

    # read params from model folder.
    model_args = read_params_from_folder(args.model_folder)

    # set device
    model_args["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get relevant keys
    argkeys = vars(args).keys()
    #print("argkeys are:", argkeys)
    for k, v in model_args.items():
        if k not in argkeys:
            setattr(args, k, v)

    if hasattr(args, 'valid_fraction'):
        setattr(args, 'train_fraction', 0.8)

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
    parser.add_argument('--sdp', type=bool, default=False,
                        help='Run SDP')
    parser.add_argument('--gurobi', type=bool, default=False,
                        help='Run Gurobi')
    parser.add_argument('--gurobi_timeout', type=float, default=5,
                        help='Timeout for Gurobi if desired')
    parser.add_argument('--greedy', type=bool, default=False,
                        help='Run greedy')

    parser.add_argument('--start_index', type=int, default=None,
                        help='Start index in dataset, for partial runs (only run i >= start_index)')
    parser.add_argument('--end_index', type=int, default=None,
                        help='End index in dataset, for partial runs (only run i < end_index)')

    parser.add_argument('--train_fraction', type=float, default=0.8,
                        help='Fraction of data to retain for training. Remainder goes to validation/testing.')

    # TODO argument to control list of baselines to run?

    args = parser.parse_args()
    modify_baseline_args(args)
    return args
