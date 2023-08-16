import torch
from model.parsing import parse_train_args
import json
import os
from data.loader import construct_loader
from model.training import train
from model.models import construct_model

# parse args
args = parse_train_args()
torch.manual_seed(args.seed)
os.makedirs(args.log_dir, exist_ok = True)


# get data, model
train_loader, val_loader = construct_loader(args)
model = construct_model(args)

# train model
train(model, train_loader)

# save params
args.device = str(args.device)
json.dump(vars(args), open(os.path.join(args.log_dir, 'params.txt'), 'w'))