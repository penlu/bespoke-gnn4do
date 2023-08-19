
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
import pickle
from datetime import datetime

'''
python test.py --model_folder="/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/_2023-08-19_09:22:09" \
    --model_file=model_ep0.pt
'''

if __name__ == '__main__':
    args = parse_test_args()

    # get data, model
    test_loader = construct_loaders(args, mode="test")
    model, _ = construct_model(args)
    criterion = get_loss_fn(args)

    # load model
    load_model(model, os.path.join(args.model_folder, args.model_file))

    # call test model
    predictions = predict(model, test_loader, args)

    # TODO: fix output file?
    pickle.dump(model, open(os.path.join(args.model_folder, 
                                        f'test_results_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.pkl'), "wb"))

    print("finished predicting!")
    
