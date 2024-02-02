
# loads model and runs it on data. 

import torch
from utils.parsing import parse_test_args
import json
import os
from data.loader import construct_loaders, test_datasets
from model.training import validate
from model.models import construct_model
from model.saving import load_model
import pickle
from datetime import datetime
import numpy as np
from problem.baselines import random_hyperplane_projector
from model.training import featurize_batch
import time
from problem.problems import get_problem

from data.gset import load_gset

'''
python test.py --model_folder="/home/bcjexu/maxcut-80/bespoke-gnn4do/training_runs/230928_runs/230925_generated_liftMP_cut/paramhash:5ec32a71d1ff22fe501f860a672a8357b01df6f08a3406ab1ae315f0ed36b69a/" \
    --model_file=best_model.pt --test_prefix=240106_TEST --problem_type=max_cut

Will load the dataset and parameters from the params in the model folder.
'''


def time_and_scores(args, model, test_loader, problem, stop_early=False):
    total_loss = 0.
    total_count = 0    
    times = []
    scores = []
    with torch.no_grad():
        for batch in test_loader:
            start_time = time.time()

            x_projs = []
            for attempt in range(10):
                x_in, batch = featurize_batch(args, batch)

                x_out = model(x_in, batch)
                loss = problem.loss(x_out, batch)

                total_loss += float(loss)

                x_proj = random_hyperplane_projector(args, x_out, batch, problem.score)

                # ENSURE we are getting a +/- 1 vector out by replacing 0 with 1
                x_proj = torch.where(x_proj == 0, 1, x_proj)
                num_zeros = (x_proj == 0).count_nonzero()
                assert num_zeros == 0

                x_projs.append(x_proj)

            end_time = time.time()

            # append times
            times.append(end_time - start_time)

            # count the score
            for i in range(len(batch)):
                attempt_scores = []
                for attempt in range(10):
                    example_x_proj = x_projs[attempt][batch.ptr[i]:batch.ptr[i + 1]]
                    example = batch.get_example(i)
                    example.penalty = args.penalty
                    score = problem.score(args, example_x_proj, example)
                    attempt_scores.append(float(score))
                print(max(attempt_scores))
                scores.append(max(attempt_scores))

            total_count += len(batch)

            if stop_early:
                return scores, times

    return scores, times

if __name__ == '__main__':
    args = parse_test_args()
    print(args)
    torch.manual_seed(args.seed)

    # get data, model
    if args.dataset not in test_datasets:
        if args.use_val_set:
            _, test_loader, _ = construct_loaders(args)
        else:
            _, _, test_loader = construct_loaders(args)
    else:
        test_loader = construct_loaders(args, mode="test")

    model, _ = construct_model(args)
    problem = get_problem(args)

    # load model
    model = load_model(model, os.path.join(args.model_folder, args.model_file))
    model.to(args.device)

    # call test model
    predictions = time_and_scores(args, model, test_loader, problem, stop_early=True) # to initialize CUDA
    predictions = time_and_scores(args, model, test_loader, problem)
    scores, times = predictions
    print(f'score mean: {sum(scores) / len(scores)}')
    print(f'score variance: {np.var(scores)}')
    print(f'total time: {sum(times)}')

    # TODO: fix output file?
    np.save(os.path.join(args.model_folder, f'{args.test_prefix}@@test_results_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.npy'), np.array(predictions))

    print("finished predicting!")
