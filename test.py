
# loads model and runs it on data. 

import torch
from utils.parsing import parse_test_args
import json
import os
from data.loader import construct_loaders
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
            if len(batch) == 1:
                datalist = [batch]
            else:
                datalist = batch.to_data_list()
            for example in datalist:
                start_time = time.time()
                x_in, example = featurize_batch(args, example)
                x_out = model(x_in, example)
                loss = problem.loss(x_out, example)

                total_loss += loss.cpu().detach().numpy()

                x_proj = random_hyperplane_projector(args, x_out, example, problem.score)
                end_time = time.time()

                # append times
                times.append(end_time - start_time)

                # ENSURE we are getting a +/- 1 vector out by replacing 0 with 1
                x_proj = torch.where(x_proj == 0, 1, x_proj)

                num_zeros = (x_proj == 0).count_nonzero()
                assert num_zeros == 0

                # count the score
                score = problem.score(args, x_proj, example)
                scores.append( float(score.cpu().detach().numpy()))
                total_count += 1

            if stop_early:
                return scores, times
                
    return scores, times


if __name__ == '__main__':
    args = parse_test_args()
    print(args)

    # get data, model
    if args.use_val_set:
        _, test_loader, _ = construct_loaders(args)
    else:
        _, _, test_loader = construct_loaders(args)
    model, _ = construct_model(args)
    problem = get_problem(args)

    # load model
    model = load_model(model, os.path.join(args.model_folder, args.model_file))
    model.to(args.device)

    # call test model
    #predictions = validate(args, model, test_loader)
    predictions = time_and_scores(args, model, test_loader, problem, stop_early=True) # to initialize CUDA
    predictions = time_and_scores(args, model, test_loader, problem)
    times, scores = predictions
    print(f'average score: {sum(scores) / len(scores)}')

    # TODO: fix output file?
    np.save(os.path.join(args.model_folder, f'{args.test_prefix}@@test_results_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.npy'), np.array(predictions))

    print("finished predicting!")
