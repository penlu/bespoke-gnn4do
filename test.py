
# loads model and runs it on data. 

import torch
from model.parsing import parse_test_args
import json
import os
from data.loader import construct_loaders
from model.training import predict
from model.models import construct_model
from model.losses import get_loss_fn
from model.saving import load_model

if __name__ == '__main__':
    args = parse_test_args()

    # TODO: read params from model folder.

    # TODO: fix this get data, model
    train_loader, val_loader = construct_loaders(args)
    model, optimizer = construct_model(args)
    criterion = get_loss_fn(args)

    # load model
    load_model(model, os.path.join(args.model_folder, args.model_file))

    # TODO: call test model
    predictions = predict(model, test_loader, args)
    # TODO: save predictions
    
