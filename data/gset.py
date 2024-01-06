# Adapted from read_gset.ipynb by Nikos
import os
from os import listdir
from os.path import isfile, join
import networkx as nx
import numpy as np
import torch_geometric
from torch_geometric.utils import from_networkx
import torch
import re

#function for reading the gset files
def load_mtx(path):
    with open(path, 'r') as f:
        g = nx.Graph()
        weights = []
        first_line = True
        for line in f:
            if not line[0] == '%':
                s = line.split()
                if first_line:
                    g.add_nodes_from(range(int(s[0])))
                    first_line = False
                else:
                    g.add_edge(int(s[0]) - 1, int(s[1]) - 1)
                    if len(s) > 2:
                        weights.append(int(s[2]))
    if len(weights) < g.number_of_edges():
        weights = np.ones(g.number_of_edges())
    else:
        print("WEIGHTED", path)
        weights = np.int64(weights)
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
        if len(tokens)==2 and 'mtx' in tokens[1]:
            counter+=1
            g, weights = load_mtx(gset_path+ '/'+file)
            graphs_and_weights[tokens[0]]=[g,weights]
        for token in tokens:
            if 'txt' in token:
                with open(gset_path+ '/'+file, mode='r',  encoding='utf-8-sig') as f:
                    lines = f.readlines()
                    for chunk in lines:
                        splitline = chunk.split('\t')
                        if not 'G' in chunk:
                            notgs+=1
                            optimal = splitline[1].split('(')[0]
                            optimal = optimal.replace(".","")
                            optimal = optimal.replace(",","")
                            optimal_values+=[optimal]
                        else:
                            notgs+=1
                            currg = splitline[0]
                            gs+=[currg]
                        parts = re.split('\n | \t', chunk)
                        
    best_known_gs = {}
    for g,opt in zip(gs,optimal_values):
        best_known_gs[g]=opt

    for g in graphs_and_weights.keys():
        graphs_and_weights[g]+=[best_known_gs[g]]

    pyg_dataset = []
    for graph in graphs_and_weights.keys():
        pyg_graph = from_networkx(graphs_and_weights[graph][0])
        pyg_graph.weights = torch.FloatTensor(graphs_and_weights[graph][1])
        print(graphs_and_weights[graph][2])
        pyg_graph.optimal = torch.FloatTensor(int(graphs_and_weights[graph][2]))
        pyg_graph.name = graph
        pyg_dataset+=[pyg_graph]

    return pyg_dataset
