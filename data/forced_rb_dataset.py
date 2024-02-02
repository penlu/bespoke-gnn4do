import itertools
from itertools import product
import random
import numpy as np
import networkx as nx
from torch_geometric.utils.convert import from_networkx

# Adapted from Nikos's Forced_RB.ipynb

## RB MODEL for hard independent set instances: 
#1) Create n disjoint cliques
#2) Pick any pair of those disjoint cliques and sample s amount of edges between them
#3) Repeat step 2 x iterations
#4) The maximum independent set will be at most n. 
#5) The is_forced argument enforces the size of the maximum independent set to be n. This can slow down the generation process slightly.
#To force the independent set to be exactly n we can create a set of nodes that contains exactly one node from each disjoint clique, and then ensure that no edge gets sampled between them.

def RB_model(generator=np.random.default_rng(0), n_range=[10, 26], k_range=[5, 21], p_range=[0.3, 1.0], is_forced=True):
    n = generator.integers(n_range[0], n_range[1]) #number of disjoint cliques (and upper bound on max independent set), feel free to change the ranges depending on comp budget or other considerations
    k = generator.integers(k_range[0], k_range[1])  #number of nodes on each disjoint clique, feel free to change as well

    p = generator.uniform(p_range[0], p_range[1]) #determines how dense the sampling will be
    a = np.log(k) / np.log(n) #parameter that determines how many edges we sample
    r = - a / np.log(1 - p)
    v = k * n #total number of nodes in the graph
    s = int(p * (n ** (2 * a)))
    iterations = int(r * n * np.log(n) - 1)


    parts = np.reshape(np.int64(range(v)), (n, k))
    total_edges = set()
    for i in parts:
        total_edges |= set(itertools.combinations(i, 2))

    edges = set()
    for _ in range(iterations):
        i, j = generator.choice(n, 2, replace=False)
        all = set(itertools.product(parts[i, :], parts[j, :]))
        all -= edges

        edges |= set(map(tuple, generator.choice(tuple(all), min(s, len(all)), replace=False)))
        total_edges |= edges

    if is_forced: ##this enforces a maximum independent set of size n py picking a random node from each disjoint clique, and then removing all the edges between all the nodes that have been picked. 
        indset = []
        for part in parts:
            indset_node = generator.choice(len(part),1)
            indset +=[part[indset_node].item()]
        forbidden = itertools.combinations(indset, 2)
        for edge in forbidden:
            while edge in total_edges:
                total_edges.remove(edge)
            while tuple(reversed(edge)) in total_edges:
                total_edges.remove(tuple(reversed(edge)))

    G = nx.Graph()
    G.add_edges_from(list(total_edges))
    G = G.to_undirected()
    return G, n

def forced_rb_generator(seed, n_min, n_max, k_min, k_max, p_min, p_max):
    generator = np.random.default_rng(seed)
    while True:
        G, n = RB_model(generator=generator, n_range=[n_min, n_max], k_range=[k_min, k_max], p_range=[p_min, p_max])
        data = from_networkx(G)
        data.optimal = n
        yield data
