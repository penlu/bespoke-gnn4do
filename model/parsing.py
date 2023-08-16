# parse command line
# create folders and stuff
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
    parser.add_argument("--TUdataset_name", type=str, default="PROTEINS", help="Path to data file")
    parser.add_argument(
        "--problem_type",
        type=str,
        default='maxcut',
        choices=['maxcut', 'vertex_cover', 'maxclique'],
        help="what problem are we doing",
    )
    parser.add_argument('--seed', type=str, default=0,
                        help='Torch random seed to use to initialize networks')

    # TODO: add relevant arguments.

def modify_train_args(args: Namespace):
    """
    Modifies and validates training arguments in place.

    :param args: Arguments.
    """
    print("device", torch.cuda.is_available())
    setattr(
        args, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    # TODO: add real log here
    args.log_dir = "training_runs/" + datetime.now().strftime("%H:%M:%S")


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

