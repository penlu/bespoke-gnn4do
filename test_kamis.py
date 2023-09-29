
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
from torch_geometric.loader import DataLoader
import pickle



'''
python test_kamis.py --model_folder="/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/230928_runs/230925_generated_preset_VC/paramhash:cc6823e1ed3cfabe4e6da1f22a84cb608a6664dff4d82038671b2487b9e9a6d6" \
    --model_file=model_step20000.pt --test_prefix=kamis_timing

Will load the dataset and parameters from the params in the model folder.
'''
from model.losses import get_loss_fn, get_score_fn
from utils.baselines import random_hyperplane_projector
from model.training import featurize_batch
import time


def time_and_scores(args, model, val_loader, criterion=None, stop_early=False):
    loss_fn = get_loss_fn(args)
    score_fn = get_score_fn(args)

    total_loss = 0.
    total_score = 0.
    total_count = 0
    times = []
    scores = []
    with torch.no_grad():
        for batch in val_loader:
            for example in batch.to_data_list():

                start_time = time.time()
                x_in, edge_index, edge_weight, node_weight = featurize_batch(args, example)
                x_out = model(
                  x=x_in,
                  edge_index=edge_index,
                  edge_weight=edge_weight,
                  node_weight=node_weight,
                  vc_penalty=args.vc_penalty
                )
                loss = loss_fn(x_out, edge_index)
                total_loss += loss.cpu().detach().numpy()

                x_proj = random_hyperplane_projector(args, x_out, example, score_fn)
                end_time = time.time()

                # append times
                times.append(end_time - start_time)

                # ENSURE we are getting a +/- 1 vector out by replacing 0 with 1
                x_proj = torch.where(x_proj == 0, 1, x_proj)

                num_zeros = (x_proj == 0).count_nonzero()
                assert num_zeros == 0

                # count the score
                score = score_fn(args, x_proj, example)
                scores.append( float(score.cpu().detach().numpy()))

                if stop_early:
                    break

    return times, scores

if __name__ == '__main__':
    args = parse_test_args()
    print(args)

    # get data, model
    #_, _, test_loader = construct_loaders(args)
    pickled_data = pickle.load(open('/home/bcjexu/maxcut-80/bespoke-gnn4do/graphs_and_results.pickle', 'rb'))
    dataset_names = ['er', 'ba', 'hk', 'ws']

    model, _ = construct_model(args)
    criterion = get_loss_fn(args)
    # load model
    model = load_model(model, os.path.join(args.model_folder, args.model_file))
    model.to(args.device)
    for ds in dataset_names:
        #ds_keys = list(pickled_data[ds].keys())
        datapoints = [y[0] for y in pickled_data[ds].values()]

        test_loader = DataLoader(datapoints, batch_size=args.batch_size, shuffle=False)

        # call test model
        #predictions = validate(args, model, test_loader)
        predictions = time_and_scores(args, model, test_loader, stop_early=True)
        predictions = time_and_scores(args, model, test_loader)
        print(predictions)

        # TODO: fix output file?
        np.save(os.path.join(args.model_folder, f'{args.test_prefix}_{ds}@@test_results_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.npy'), np.array(predictions))

        print("finished predicting!")
