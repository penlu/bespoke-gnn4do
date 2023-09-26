import json
import os

import torch

from data.loader import construct_loaders
from model.losses import get_loss_fn
from model.models import construct_model
from model.parsing import parse_train_args
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
    criterion = get_loss_fn(args)

    # train model
    train(args, model, train_loader, optimizer, criterion, val_loader=val_loader, test_loader=test_loader)

    # run final validation
    valid_loss = validate(args, model, val_loader, criterion)

    # write "done" file
    with open(os.path.join(args.log_dir, 'done.txt'), 'w') as file:
        file.write(f"{valid_loss}\n")
        file.write("done.\n")
