# Adapted from read_gset.ipynb by Nikos
import os
from os import listdir
from os.path import isfile, join
import networkx as nx
import numpy as np
import torch_geometric
from torch_geometric.utils import from_networkx, coalesce
from torch_geometric.data import Data, Batch
import torch
import re

#function for reading the gset files
def load_mtx(path):
    with open(path, 'r') as f:
        g = [[], []]
        weights = []
        first_line = True
        for line in f:
            if line[0] == '%':
                continue

            if first_line:
                first_line = False
                continue

            s = line.split()
            g[0] += [int(s[0]) - 1, int(s[1]) - 1]
            g[1] += [int(s[1]) - 1, int(s[0]) - 1]
            if len(s) > 2:
                weights.append(int(s[2]))
                weights.append(int(s[2]))

    if len(weights) == 0:
        weights = np.ones(len(g[0]))

    return g, weights

def load_gset(gset_path):
    #Prepping the dataset
    graphs_and_weights = {}
    optimal_values = []
    counter = 0
    notgs = 0
    gs = []
    for file in listdir(gset_path):
        tokens = file.split('.')
        if len(tokens) == 2 and 'mtx' in tokens[1]:
            counter += 1
            edge_index, edge_weight = load_mtx(gset_path+ '/'+file)
            graphs_and_weights[tokens[0]] = [edge_index, edge_weight]

        for token in tokens:
            if 'txt' in token:
                with open(gset_path+'/'+file, mode='r', encoding='utf-8-sig') as f:
                    lines = f.readlines()
                    for chunk in lines:
                        splitline = chunk.split('\t')
                        if not 'G' in chunk:
                            notgs += 1
                            optimal = splitline[1].split('(')[0]
                            optimal = optimal.replace(".","")
                            optimal = optimal.replace(",","")
                            optimal_values+=[optimal]
                        else:
                            notgs += 1
                            currg = splitline[0]
                            gs += [currg]
                        parts = re.split('\n | \t', chunk)

    best_known_gs = {}
    for g, opt in zip(gs, optimal_values):
        best_known_gs[g] = opt

    for g in graphs_and_weights.keys():
        graphs_and_weights[g] += [best_known_gs[g]]

    # at this point, graphs_and_weights is a dict from names (e.g. "G14") to:
    # a tuple of [edge_index, edge_weight], and possibly a third item representing optimal score

    pyg_dataset = []
    for graph in graphs_and_weights.keys():
        edge_index=torch.LongTensor(graphs_and_weights[graph][0])
        edge_weight=torch.FloatTensor(graphs_and_weights[graph][1])

        edge_index, edge_weight = coalesce(edge_index, edge_weight)

        pyg_graph = Data(
            edge_index=edge_index,
            edge_weight=edge_weight,
            optimal=int(graphs_and_weights[graph][2]),
            name=graph)
        pyg_dataset += [pyg_graph]

    return pyg_dataset
