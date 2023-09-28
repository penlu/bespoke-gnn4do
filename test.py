
# loads model and runs it on data. 

import torch
from model.parsing import parse_test_args
import json
import os
from data.loader import construct_loaders
from model.training import validate
from model.models import construct_model
from model.losses import get_loss_fn
from model.saving import load_model
import pickle
from datetime import datetime
import numpy as np

'''
python test.py --model_folder="/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/230924_hparam/paramhash:0a0656a369a5b8e4a4be27e0d04fb3b8c161e7b630caf99b8eaeedcddd6a2b18" \
    --model_file=best_model.pt --problem_type=vertex_cover --dataset=ENZYMES
'''

if __name__ == '__main__':
    args = parse_test_args()

    # get data, model
    _, _, test_loader = construct_loaders(args)
    model, _ = construct_model(args)
    criterion = get_loss_fn(args)

    # load model
    model = load_model(model, os.path.join(args.model_folder, args.model_file))
    model.to(args.device)

    # call test model
    predictions = validate(args, model, test_loader) #predict(model, test_loader, args)
    print(predictions)

    # TODO: fix output file?
    np.save(os.path.join(args.model_folder, f'{args.prefix}@@test_results_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.np'), np.array(predictions))

    print("finished predicting!")
    
