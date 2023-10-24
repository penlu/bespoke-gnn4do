import json
import os

import torch

from data.loader import construct_loaders
from problem.problems import get_problem
from model.models import construct_model
from utils.parsing import parse_train_args
from model.training import train, validate

if __name__ == '__main__':
    # parse args
    args = parse_train_args()
    print(args)
    torch.manual_seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)

    # save params
    args.device = str(args.device)
    json.dump(vars(args), open(os.path.join(args.log_dir, 'params.txt'), 'w'))

    # get data, model
    train_loader, val_loader, test_loader = construct_loaders(args)
    model, optimizer = construct_model(args)
    problem = get_problem(args)

    # train model
    train(args, model, train_loader, optimizer, problem, val_loader=val_loader, test_loader=test_loader)

    # run final validation
    valid_loss = validate(args, model, val_loader, problem)

    # write "done" file
    with open(os.path.join(args.log_dir, 'done.txt'), 'w') as file:
        file.write(f"{valid_loss}\n")
        file.write("done.\n")
