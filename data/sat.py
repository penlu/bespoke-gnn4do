# Loading SAT problems
import itertools
import numpy as np
import torch
from torch_geometric.utils import to_edge_index
from torch_geometric.data import Data

# turn a DIMACS input into a clause-list representation
def dimacs_parser():
    pass

def dimacs_printer(N, K, clauses, signs):
    out = f"p cnf {N} {K}\n"
    signed_clauses = (clauses + 1) * signs
    for f in range(K):
        out += f"{signed_clauses[f, 0]} {signed_clauses[f, 1]} {signed_clauses[f, 2]} 0\n"
    return out

# generate clause-list representation of random 3-SAT problem
def random_3sat_clauses(random_state, N=100, K=50, p=0.5):
    # generate random clauses
    clauses = random_state.randint(0, N, (K, 3))

    # put clauses in variable order
    clauses = np.sort(clauses, axis=-1)

    # generate random sign: -1 means corresponding var appears in clause inverted
    signs = random_state.binomial(1, p, size=(K, 3)) * 2 - 1

    return clauses, signs

# constraints of the form <x_i, x_j> = <x_ij, e1>
def make_undirected_constraint(total_vars, pair_to_index, i, j):
    #Ci = []; Cv = []
    #Ci.append([i+1, j+1]); Cv.append(1.)
    #Ci.append([pair_to_index[(i, j)]+1, 0]); Cv.append(-1.)
    #C = torch.sparse_coo_tensor(indices=list(zip(*Ci)), values=Cv, size=(total_vars+1, total_vars+1))
    return [i+1, j+1, pair_to_index[(i, j)]+1, 0]
    #return C

# constraints of the form <x_i, x_ij> = <x_j, e1>
def make_directed_constraint(total_vars, pair_to_index, i, j):
    a, b = sorted((i, j))
    #Ci = []; Cv = []
    #Ci.append([i+1, pair_to_index[(a, b)]+1]); Cv.append(1.)
    #Ci.append([j+1, 0]); Cv.append(-1.)
    #C = torch.sparse_coo_tensor(indices=list(zip(*Ci)), values=Cv, size=(total_vars+1, total_vars+1))
    return [i+1, pair_to_index[(a, b)]+1, j+1, 0]
    #return C

# constraints of the form <x_i, x_jk> = <x_ij, x_k>
def make_triangle_constraint(total_vars, pair_to_index, i, j, k):
    #Ci = []; Cv = []
    #Ci.append([i+1, pair_to_index[(j, k)]+1]); Cv.append(1.)
    #Ci.append([j+1, pair_to_index[(i, k)]+1]); Cv.append(-1.)
    #C1 = torch.sparse_coo_tensor(indices=list(zip(*Ci)), values=Cv, size=(total_vars+1, total_vars+1))
    C1 = [i+1, pair_to_index[(j, k)]+1, j+1, pair_to_index[(i, k)]+1]

    #Ci = []; Cv = []
    #Ci.append([j+1, pair_to_index[(i, k)]+1]); Cv.append(1.)
    #Ci.append([k+1, pair_to_index[(i, j)]+1]); Cv.append(-1.)
    #C2 = torch.sparse_coo_tensor(indices=list(zip(*Ci)), values=Cv, size=(total_vars+1, total_vars+1))
    C2 = [j+1, pair_to_index[(i, k)]+1, k+1, pair_to_index[(i, j)]+1]

    #Ci = []; Cv = []
    #Ci.append([i+1, pair_to_index[(j, k)]+1]); Cv.append(1.)
    #Ci.append([k+1, pair_to_index[(i, j)]+1]); Cv.append(-1.)
    #C3 = torch.sparse_coo_tensor(indices=list(zip(*Ci)), values=Cv, size=(total_vars+1, total_vars+1))
    C3 = [i+1, pair_to_index[(j, k)]+1, k+1, pair_to_index[(i, j)]+1]

    return C1, C2, C3

# constraints of the form <x_ij, x_jk> = <x_i, x_k>
def make_quad_constraint(total_vars, pair_to_index, i, j, k):
    a, b = sorted((i, j))
    c, d = sorted((j, k))
    e, f = sorted((i, k))

    #Ci = []; Cv = []
    #Ci.append([pair_to_index[(a, b)]+1, pair_to_index[(c, d)]+1]); Cv.append(1.)
    #Ci.append([e+1, f+1]); Cv.append(-1.)
    #C = torch.sparse_coo_tensor(indices=list(zip(*Ci)), values=Cv, size=(total_vars+1, total_vars+1))

    return [pair_to_index[(a, b)]+1, pair_to_index[(c, d)]+1, e+1, f+1]

    #return C

# turn SAT problem into A, C sparse tensors, then emplace in pytorch geometric Data object
def compile_sat(clauses, signs, N, K):
    # obtain the pair variable dictionary, assigning an index to each pair
    pair_to_index = {}
    total_vars = N
    for f in range(K):
        # go over each pair of vars in this clause
        for v1, v2 in itertools.combinations(list(clauses[f]), 2):
            i, j = tuple(sorted((int(v1), int(v2))))
            if (i, j) not in pair_to_index:
                # assign new variable
                pair_to_index[(i, j)] = total_vars
                total_vars += 1

    #print(f"total vars: {total_vars}")
    #print(f"pair_to_index: {pair_to_index}")

    # compile the SAT problem to A and C sparse tensors
    As = [] # list of weight matrices
    Cs = [] # list of constraint matrices
    for f in range(K):
        i = int(clauses[f, 0])
        j = int(clauses[f, 1])
        k = int(clauses[f, 2])
        tau_i = signs[f, 0]
        tau_j = signs[f, 1]
        tau_k = signs[f, 2]
        weight = 1.

        # CONSTRUCT A TERMS FOR THIS CLAUSE
        Ai = []
        Av = []

        # order 1 entries
        Ai.append([i+1, 0]); Av.append(tau_i)
        Ai.append([j+1, 0]); Av.append(tau_j)
        Ai.append([k+1, 0]); Av.append(tau_k)

        # order 2 entries
        Ai.append([i+1, j+1]); Av.append(-tau_i * tau_j)
        Ai.append([i+1, k+1]); Av.append(-tau_i * tau_k)
        Ai.append([j+1, k+1]); Av.append(-tau_j * tau_k)

        # order 3 entries
        Ai.append([pair_to_index[(i, j)]+1, k+1]); Av.append(tau_i * tau_j * tau_k / 3.)
        Ai.append([pair_to_index[(i, k)]+1, j+1]); Av.append(tau_i * tau_j * tau_k / 3.)
        Ai.append([pair_to_index[(j, k)]+1, i+1]); Av.append(tau_i * tau_j * tau_k / 3.)

        # the constant term
        Ai.append([0, 0]); Av.append(7.)

        A = torch.sparse_coo_tensor(indices=list(zip(*Ai)), values=Av, size=(total_vars+1, total_vars+1))
        As.append(A * weight / 8.)

        # ADD C TERMS FOR THIS CLAUSE
        # undirected constraints
        Cs.append(make_undirected_constraint(total_vars, pair_to_index, i, j))
        Cs.append(make_undirected_constraint(total_vars, pair_to_index, j, k))
        Cs.append(make_undirected_constraint(total_vars, pair_to_index, i, k))

        # directed constraints
        Cs.append(make_directed_constraint(total_vars, pair_to_index, i, j))
        Cs.append(make_directed_constraint(total_vars, pair_to_index, j, i))
        Cs.append(make_directed_constraint(total_vars, pair_to_index, i, k))
        Cs.append(make_directed_constraint(total_vars, pair_to_index, k, i))
        Cs.append(make_directed_constraint(total_vars, pair_to_index, j, k))
        Cs.append(make_directed_constraint(total_vars, pair_to_index, k, j))

        # triangle constraints
        Cs += make_triangle_constraint(total_vars, pair_to_index, i, j, k)

        # quad constraints
        Cs.append(make_quad_constraint(total_vars, pair_to_index, i, j, k))
        Cs.append(make_quad_constraint(total_vars, pair_to_index, i, k, j))
        Cs.append(make_quad_constraint(total_vars, pair_to_index, j, i, k))

    A = torch.sparse.sum(torch.stack(As), dim=0).float()
    C = torch.LongTensor(Cs)

    return total_vars, pair_to_index, A, C

def random_3sat_generator(seed, n=100, K=400, p=0.5):
    if isinstance(n, int):
        N = n
        n_max = n
    elif isinstance(n, list) and len(n) == 1:
        N = n[0]
        n_max = n[0]
    elif isinstance(n, list) and len(n) == 2:
        N, n_max = n
    else:
        raise ValueError('random_3sat_generator got bad n (expected int, [n], or [n_min, n_max]: {n}')

    random_state = np.random.RandomState(seed)
    while True:
        # generate A and C for a random 3-SAT instance
        clauses, signs = random_3sat_clauses(random_state, N, K, p)
        total_vars, pair_to_index, A, C = compile_sat(clauses, signs, N, K)

        # represent these as a Data object
        edge_list, edge_weight = to_edge_index(A)
        decremented = edge_list - 1

        negative_columns = (decremented < 0).any(dim=0)
        keep_columns = ~negative_columns
        edge_index = decremented[:, keep_columns]
        edge_weight = edge_weight[keep_columns]

        yield Data(
            num_nodes=total_vars,
            edge_index=edge_index,
            edge_weight=edge_weight,
            clauses=clauses, signs=signs,
            A=A, C=C, N=N, K=K,
            pair_to_index=pair_to_index)

# produces the number of satisfied clauses
def sat_objective(X, A, C, N, K):
    # expand X to include e1
    X = torch.cat([torch.zeros(1, X.shape[1], device=X.device), X], dim=0)
    X[0, 0] = 1.

    # calculate objective
    XX = torch.matmul(X, torch.transpose(X, 0, 1))
    objective = torch.trace(torch.matmul(A, XX))

    # calculate penalties
    x1_i = X[C[:, 0]]
    x1_j = X[C[:, 1]]
    x2_i = X[C[:, 2]]
    x2_j = X[C[:, 3]]

    X1 = torch.sum(x1_i * x1_j, dim=1)
    X2 = torch.sum(x2_i * x2_j, dim=1)

    penalties = X2 - X1

    return -objective, torch.sum(penalties * penalties)

# count number of satisfied clauses
def count_sat_clauses(X, clauses, signs):
    count = 0
    for f in range(len(clauses)):
        i = int(clauses[f, 0])
        j = int(clauses[f, 1])
        k = int(clauses[f, 2])
        tau_i = signs[f, 0]
        tau_j = signs[f, 1]
        tau_k = signs[f, 2]

        if tau_i * X[i] == 1 or tau_j * X[j] == 1 or tau_k * X[k] == 1:
            count += 1

    return count

# check count_sat_clauses(clauses, signs, assignment) == sat_objective(compile_sat(clauses, signs, ...), vectorize(assignment))
# TODO test penalties by perturbing doubles vars
# TODO get average scores for a random assignment
def run_equivalence_test(clauses, signs, X):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_vars, pair_to_index, A, C = compile_sat(clauses, signs, N, K)
    A = A.to(device)
    C = C.to(device)

    # extend random assignment into ij domain
    X_ext = torch.zeros(total_vars, 1, device=device)
    for i in range(N):
        X_ext[i, 0] = X[i]

    for f in range(K):
        i = int(clauses[f, 0])
        j = int(clauses[f, 1])
        k = int(clauses[f, 2])
        X_ext[pair_to_index[(i, j)], 0] = X[i] * X[j]
        X_ext[pair_to_index[(i, k)], 0] = X[i] * X[k]
        X_ext[pair_to_index[(j, k)], 0] = X[j] * X[k]
    objective, penalty = sat_objective(X_ext, A, C, N, K)
    print(count_sat_clauses(X, clauses, signs))
    print(objective)
    print(C.shape)
    print("should be zero:", penalty)

if __name__ == '__main__':
    print("hello! I am data/sat.py and it is time to run some tests")
    seed = 0
    N = 50
    K = 200
    random_state = np.random.RandomState(seed)
    clauses, signs = random_3sat_clauses(random_state, N, K)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.bernoulli(torch.full((N,), 0.5, device=device))*2-1

    # run test: produce random assignment, then feed to both
    run_equivalence_test(clauses, signs, X)
