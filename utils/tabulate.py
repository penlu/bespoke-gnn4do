import os
import json
import pandas as pd
import numpy as np
import sys
from pathlib import Path

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from data.loader import RandomGraphDataset

# Utils for making pandas dataframe tables

def load_datasets():
    datasets = {}
    datasets['RANDOM'] = RandomGraphDataset(
      root='/tmp/random',
      num_graphs=1000,
      num_nodes_per_graph=100,
      edge_probability=0.15,
    )
    for dataset in ['PROTEINS', 'ENZYMES', 'COLLAB', 'IMDB-BINARY', 'MUTAG']:
        loader = TUDataset(root=f'/tmp/{dataset}', name=dataset)
        datasets[dataset] = loader

    return datasets

# TODO vertex and edge counts for each dataset

# Collect training outputs; return map (model, dataset) -> (losses, scores)
def load_train_outputs(path, prefix):
    model_list = [path / x for x in os.listdir(path) if x.startswith(prefix)]

    # load in params
    outputs = {}
    for model_folder in model_list:
        try:
            with open(os.path.join(model_folder, 'params.txt'), 'r') as f:
                model_args = json.load(f)
            train_losses = np.load(os.path.join(model_folder, 'train_losses.npy'))
            valid_scores = np.load(os.path.join(model_folder, 'valid_scores.npy'))
            if model_args['dataset'] == 'TU':
                dataset = model_args['TUdataset_name']
            else:
                dataset = 'RANDOM'

            outputs[(model_args['model_type'], dataset)] = (train_losses, valid_scores)
            print(f"load_train_outputs: got {model_args['model_type']}, {dataset}")
        except:
            print(f'load_train_outputs: problem with {model_folder}')
            print(sys.exc_info())

    return outputs

# Collect baseline outputs
def load_baseline_outputs(path, prefix, method, indices=None):
    folder_list = [path / x for x in os.listdir(path) if x.startswith(prefix)]

    # load in params
    outputs = {}
    for folder in folder_list:
        try:
            with open(folder / 'params.txt', 'r') as f:
                args = json.load(f)
            if args['dataset'] == 'TU':
                dataset = args['TUdataset_name']
            else:
                dataset = "RANDOM"

            with open(folder / 'results.jsonl', 'r') as f:
                total_score = 0.
                count = 0
                for line in f:
                    res = json.loads(line)
                    # second condition is: only do this if the graph is in the validation set
                    if res['method'] == method and (indices == None or res['index'] in indices[dataset]):
                        total_score += res['score']
                        count += 1

            outputs[dataset] = total_score / count

            print(f"load_baseline_outputs: {dataset} length: {count}")
            print(f"load_baseline_outputs: {dataset} {method}: {total_score / count}")
        except:
            print(f'something is wrong w/ {folder}')
            print(sys.exc_info())

    return outputs
