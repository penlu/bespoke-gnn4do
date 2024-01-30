# Loading SAT problems
import itertools
import numpy as np
import torch
from torch_geometric.utils import to_edge_index
from torch_geometric.data import Data, Batch

class SDPCompiler():
    def __init__(self, N):
        self.N = N
        self.constant = 0.
        self.linear = {}
        self.quadratic = {}
        self.constraints3 = []
        self.constraints4 = []

    def add_constant(self, v):
        self.constant += v

    def add_linear(self, i, v):
        self.linear[i] = self.linear.get(i, 0.) + v

    def add_quadratic(self, i, j, v):
        self.quadratic[(i, j)] = self.quadratic.get((i, j), 0.) + v / 2.
        self.quadratic[(j, i)] = self.quadratic.get((j, i), 0.) + v / 2.

    # only supporting constraints of form <e1, i> = <j, k>
    def add_constraint3(self, i, j, k):
        self.constraints3.append([i, j, k])

    # only supporting constraints of form <i1, j1> = <i2, j2>
    def add_constraint4(self, i1, j1, i2, j2):
        self.constraints4.append([i1, j1, i2, j2])

    # returns:
    # A0 -- constant term
    # A1i -- linear term indices
    # A1w -- linear term weights
    # A2i -- quadratic term indices -- 2 x E
    # A2w -- quadratic term weights -- E
    # C3 -- constraint indices -- 3 x something
    # C4 -- constraint indices -- 4 x something
    def compile(self):
        A0 = self.constant
        A1i = torch.LongTensor(list(self.linear.keys()))
        A1w = torch.FloatTensor(list(self.linear.values()))
        A2i = torch.LongTensor(list(self.quadratic.keys()))
        A2w = torch.FloatTensor(list(self.quadratic.values()))
        C3 = torch.LongTensor(list(self.constraints3))
        C4 = torch.LongTensor(list(self.constraints4))

        return Data(
            num_nodes=self.N,
            bias_index=A1i,
            bias_weight=A1w,
            edge_index=A2i.t(),
            edge_weight=A2w,
            C3_index=C3.t(),
            C4_index=C4.t(),
            A0=A0)

def sdp_objective(X, batch):
    # constant term
    if isinstance(batch.A0, float):
        const = batch.A0
    else:
        const = torch.sum(batch.A0)

    # linear term ("bias")
    A1i = batch.bias_index
    A1w = batch.bias_weight
    linear = torch.sum(X[A1i, 0] * A1w)

    # quadratic term ("edge")
    A2i = batch.edge_index
    A2w = batch.edge_weight
    edges = torch.sum(X[A2i[0, :]] * X[A2i[1, :]], dim=1)
    quadratic = torch.sum(edges * A2w)

    objective = const + linear + quadratic
    return objective

def sdp_constraint(X, batch):
    # constraints involving e1
    C3 = batch.C3_index
    constraints3 = X[C3[0, :], 0] - torch.sum(X[C3[1, :]] * X[C3[2, :]], dim=1)

    # constraints not involving e1
    C4 = batch.C4_index
    constraints4 = torch.sum(X[C4[0, :]] * X[C4[1, :]] - X[C4[2, :]] * X[C4[3, :]], dim=1)

    constraint = torch.sum(constraints3 * constraints3) + torch.sum(constraints4 * constraints4)
    return constraint / 4.

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
def add_undirected_constraint(compiler, pair_to_index, i, j):
    compiler.add_constraint3(pair_to_index[i, j], i, j)

# constraints of the form <x_i, x_ij> = <x_j, e1>
def add_directed_constraint(compiler, pair_to_index, i, j):
    a, b = sorted((i, j))
    compiler.add_constraint3(j, i, pair_to_index[a, b])

# constraints of the form <x_i, x_jk> = <x_ij, x_k>
def add_triangle_constraints(compiler, pair_to_index, i, j, k):
    compiler.add_constraint4(i, pair_to_index[j, k], j, pair_to_index[i, k])
    compiler.add_constraint4(j, pair_to_index[i, k], k, pair_to_index[i, j])
    compiler.add_constraint4(i, pair_to_index[j, k], k, pair_to_index[i, j])

# constraints of the form <x_ij, x_jk> = <x_i, x_k>
def add_quad_constraint(compiler, pair_to_index, i, j, k):
    a, b = sorted((i, j))
    c, d = sorted((j, k))
    e, f = sorted((i, k))
    compiler.add_constraint4(e, f, pair_to_index[a, b], pair_to_index[c, d])

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
    compiler = SDPCompiler(total_vars)
    for f in range(K):
        i = int(clauses[f, 0])
        j = int(clauses[f, 1])
        k = int(clauses[f, 2])
        tau_i = signs[f, 0]
        tau_j = signs[f, 1]
        tau_k = signs[f, 2]
        weight = 1.

        # CONSTRUCT A TERMS FOR THIS CLAUSE

        # order 1 entries
        compiler.add_linear(i, tau_i * weight / 8.)
        compiler.add_linear(j, tau_j * weight / 8.)
        compiler.add_linear(k, tau_k * weight / 8.)

        # order 2 entries
        compiler.add_quadratic(i, j, -tau_i * tau_j * weight / 8.)
        compiler.add_quadratic(i, k, -tau_i * tau_k * weight / 8.)
        compiler.add_quadratic(j, k, -tau_j * tau_k * weight / 8.)

        # order 3 entries
        compiler.add_quadratic(k, pair_to_index[i, j], tau_i * tau_j * tau_k * weight / 24.)
        compiler.add_quadratic(j, pair_to_index[i, k], tau_i * tau_j * tau_k * weight / 24.)
        compiler.add_quadratic(i, pair_to_index[j, k], tau_i * tau_j * tau_k * weight / 24.)

        # the constant term
        compiler.add_constant(7./8.)

        # ADD C TERMS FOR THIS CLAUSE
        # undirected constraints
        add_undirected_constraint(compiler, pair_to_index, i, j)
        add_undirected_constraint(compiler, pair_to_index, j, k)
        add_undirected_constraint(compiler, pair_to_index, i, k)

        # directed constraints
        add_directed_constraint(compiler, pair_to_index, i, j)
        add_directed_constraint(compiler, pair_to_index, j, i)
        add_directed_constraint(compiler, pair_to_index, i, k)
        add_directed_constraint(compiler, pair_to_index, k, i)
        add_directed_constraint(compiler, pair_to_index, j, k)
        add_directed_constraint(compiler, pair_to_index, k, j)

        # triangle constraints
        add_triangle_constraints(compiler, pair_to_index, i, j, k)

        # quad constraints
        add_quad_constraint(compiler, pair_to_index, i, j, k)
        add_quad_constraint(compiler, pair_to_index, i, k, j)
        add_quad_constraint(compiler, pair_to_index, j, i, k)

    data = compiler.compile()

    # how to recompute pairs from singles
    pair_index = [[], [], []]
    for index, i in pair_to_index.items():
        pair_index[0].append(i)
        pair_index[1].append(index[0])
        pair_index[2].append(index[1])
    pair_index = torch.LongTensor(pair_index)
    data.pair_index = pair_index

    data.num_vars = N
    data.num_clauses = K
    data.clause_index = clauses
    data.signs = signs

    return total_vars, pair_to_index, data

def print_sat_data(d):
    print(f"total_vars={d.num_nodes}\nbias_index={d.bias_index}\nbias_weight={d.bias_weight}\nedge_index={d.edge_index}\nedge_weight={d.edge_weight}\nC3_index={d.C3_index}\nC4_index={d.C4_index}")

def random_3sat_generator(seed, n_min=100, n_max=100, k_min=400, k_max=400, p_min=0.5, p_max=0.5):
    random_state = np.random.RandomState(seed)
    while True:
        if n_min != n_max:
            N = random_state.randint(n_min, n_max + 1)
        else:
            N = n_min

        if k_min != k_max:
            K = random_state.randint(k_min, k_max + 1)
        else:
            K = k_min

        if p_min != p_max:
            p = random_state.uniform(p_min, p_max)
        else:
            p = p_min

        # generate A and C for a random 3-SAT instance
        clauses, signs = random_3sat_clauses(random_state, N, K, p)
        total_vars, pair_to_index, data = compile_sat(clauses, signs, N, K)

        yield data

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
    total_vars, pair_to_index, data = compile_sat(clauses, signs, N, K)
    data = data.to(device)

    # extend random assignment into ij domain
    X_ext = torch.zeros(total_vars, 1, device=device)
    for i in range(N):
        X_ext[i, 0] = X[i]

    for f in range(K):
        i = int(clauses[f, 0])
        j = int(clauses[f, 1])
        k = int(clauses[f, 2])
        X_ext[pair_to_index[i, j], 0] = X[i] * X[j]
        X_ext[pair_to_index[i, k], 0] = X[i] * X[k]
        X_ext[pair_to_index[j, k], 0] = X[j] * X[k]
    objective = sdp_objective(X_ext, data)
    constraint = sdp_constraint(X_ext, data)
    print(count_sat_clauses(X, clauses, signs))
    print(objective)
    print("should be zero:", constraint)

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

    gen = random_3sat_generator(0, n=4, K=2, p=0.5)
    d1 = gen.__next__()
    d2 = gen.__next__()
    print_sat_data(d1)
    print()
    print_sat_data(d2)
    print()

    batch = Batch.from_data_list([d1, d2])
    print_sat_data(batch)
