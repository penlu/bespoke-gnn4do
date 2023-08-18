import torch
from model.parsing import parse_train_args
import json
import os
from data.loader import construct_loaders
from model.training import train
from model.models import construct_model
from model.losses import get_loss_fn

if __name__ == '__main__':
    # parse args
    args = parse_train_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)

    # save params
    args.device = str(args.device)
    json.dump(vars(args), open(os.path.join(args.log_dir, 'params.txt'), 'w'))

    # get data, model
    train_loader, val_loader = construct_loaders(args)
    model, optimizer = construct_model(args)
    criterion = get_loss_fn(args)

    # train model
    train(args, model, train_loader, optimizer, criterion)
