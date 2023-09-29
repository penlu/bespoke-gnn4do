import pickle
import os
import numpy as np
import torch
from argparse import Namespace
from model.parsing import read_params_from_folder
from data.loader import construct_loaders
from model.training import validate
from model.models import construct_model
from model.losses import get_loss_fn
from model.saving import load_model
from model.losses import get_loss_fn, get_score_fn
from utils.baselines import random_hyperplane_projector
from model.training import featurize_batch
import time

rootfolder = '/home/penlu/code/bespoke-gnn4do'
the_path = '/home/penlu/code/bespoke-gnn4do/training_runs/230928_runs/230925_generated_liftMP_VC'

def get_validation_score(model_folder):
    # train-time validation scores
    valid_scores = np.load(os.path.join(model_folder, 'valid_scores.npy'))
    max_valid_score = max(valid_scores)

    # check whether a revalidation occurred; compute new validation score if so
    revalidation_files = [x for x in os.listdir(model_folder) if x.startswith('revalidate')]
    for file in revalidation_files:
        revalid_times, revalid_scores = np.load(os.path.join(model_folder, file))
        assert(revalid_scores.shape[0] == 1000)
        revalid_score = np.average(revalid_scores)
        if revalid_score > max_valid_score:
            max_valid_score = revalid_score
            #print(f'better revalid: {file}')

    return max_valid_score

def get_kamis_score(model_folder, dataset):
    kamis_files = [x for x in os.listdir(model_folder) if x.startswith(f'kamis_test_{dataset}')]
    for file in kamis_files:
        kamis_times, kamis_scores = np.load(os.path.join(model_folder, file))
        assert(kamis_scores.shape[0] == 500)
        return np.average(kamis_times), np.average(kamis_scores)

def load_problems():
    x = pickle.load(open(f'{rootfolder}/graphs_and_results.pickle', 'rb'))

    problems = {}

    for problem, hugedict in x.items():
        graphs = []
        k_solution_sizes = []
        k_times = []
        l_solution_sizes = []
        l_times = []
        for dataname, information in hugedict.items():
            graph, kamis, lwd = information
            assert kamis[0] == 'kamis'
            assert lwd[0] == 'lwd'
            
            N = dataname.split("_")[1]
            maybeN = dataname.split("_")[2]
            if N != maybeN: print(dataname)
            if int(N) != 500: print(dataname)

            graphs.append(graph)
            k_solution_sizes.append(int(N) - kamis[1])
            k_times.append(kamis[2])
            l_solution_sizes.append(int(N) - lwd[1])
            l_times.append(lwd[2])
        problems[problem] = dict(zip(['problem', "k_solution_sizes","l_solution_sizes", "k_times", "l_times", 'graphs'], [problem, k_solution_sizes, l_solution_sizes, k_times, l_times, graphs] ))

    return problems

if __name__ == '__main__':
    problems = load_problems()

    for model_folder in os.listdir(the_path):
        full_folder = f'{the_path}/{model_folder}'
        model_args = read_params_from_folder(full_folder)

        if model_args['dataset'] == 'ErdosRenyi' and model_args['gen_n'][1] == 500:
            graphs = problems['er']['graphs']
            validation_score = get_validation_score(full_folder)
            test_time, test_score = get_kamis_score(full_folder, 'er')
            print(f'ER valid={validation_score} test={test_score} time={test_time}')
        elif model_args['dataset'] == 'BarabasiAlbert' and model_args['gen_n'][1] == 500:
            graphs = problems['ba']['graphs']
            validation_score = get_validation_score(full_folder)
            test_time, test_score = get_kamis_score(full_folder, 'ba')
            print(f'BA valid={validation_score} test={test_score} time={test_time}')
        elif model_args['dataset'] == 'PowerlawCluster' and model_args['gen_n'][1] == 500:
            graphs = problems['hk']['graphs']
            validation_score = get_validation_score(full_folder)
            test_time, test_score = get_kamis_score(full_folder, 'hk')
            print(f'HK valid={validation_score} test={test_score} time={test_time}')
        elif model_args['dataset'] == 'WattsStrogatz' and model_args['gen_n'][1] == 500:
            graphs = problems['ws']['graphs']
            validation_score = get_validation_score(full_folder)
            test_time, test_score = get_kamis_score(full_folder, 'ws')
            print(f'WS valid={validation_score} test={test_score} time={test_time}')
