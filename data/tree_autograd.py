# Loading SAT problems
import itertools
import numpy as np
import torch
from torch_geometric.utils import to_edge_index
from torch_geometric.data import Data, Batch
from torch.optim import Adam
import time

import sys
sys.path.append('/home/gridsan/myau/bespoke-gnn4do/problem')
sys.path.append('/home/gridsan/myau/bespoke-gnn4do')
    
#from baselines import random_hyperplane_projector
#from problem.problems import SATProblem
#from problem.losses import max_cut_obj, vertex_cover_obj, vertex_cover_constraint
#from problem.losses import max_cut_score, vertex_cover_score, max_clique_score
#from networkx.algorithms.approximation import one_exchange, min_weighted_vertex_cover
#from problem.baselines import max_cut_sdp, vertex_cover_sdp
#from problem.baselines import max_cut_gurobi, vertex_cover_gurobi
from problems import SATProblem

# returns a torch.FloatTensor size (N,)
def random_hyperplane_projector(args, x_lift, example, score_fn):
    if isinstance(x_lift, np.ndarray):
        x_lift = torch.FloatTensor(x_lift)

    n_hyperplanes = 1000 # TODO make this modifiable in args
    n_groups = 1 # we do it in groups to reduce memory consumption
    x_int = []
    scores = []
    for i in range(n_groups):
        hyper = torch.randn((n_hyperplanes // n_groups, x_lift.shape[1]), device=x_lift.device)
        hyper = F.normalize(hyper)
        x_proj = torch.matmul(hyper, x_lift.t())
        group_x_int = torch.sign(x_proj)[:, :, None]
        group_scores = torch.vmap(lambda x: score_fn(args, x, example))(group_x_int)
        x_int.append(group_x_int)
        scores.append(group_scores)
    x_int = torch.cat(x_int)
    scores = torch.cat(scores)
    best = torch.argmax(scores)
    out = x_int[best, :, 0]

    num_zeros = (out == 0).count_nonzero()
    if num_zeros > 0:
        print("WARNING: detected zeros in hyperplane rounding output")

    return out

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
            penalty = 0.1,
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

def sample_unique_numbers(a, b, num_samples):
    if num_samples > b - a + 1:
        raise ValueError("Number of samples requested is greater than the size of the interval")

    selected = set()
    while len(selected) < num_samples:
        n = torch.randint(low=a, high=b + 1, size=(1,)).item()
        selected.add(n)

    return torch.tensor(list(selected))
    
def batch_autograd(clauses,signs,X):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_vars, pair_to_index, data = compile_sat(clauses, signs, N, K)
    data = data.to(device)

    # extend random assignment into ij domain
    r = 5
    X_ext = torch.zeros(total_vars, r, device=device)
    for i in range(N):
        X_ext[i, 0] = X[i]

    for f in range(K):
        i = int(clauses[f, 0])
        j = int(clauses[f, 1])
        k = int(clauses[f, 2])
        X_ext[pair_to_index[i, j], 0] = X[i] * X[j]
        X_ext[pair_to_index[i, k], 0] = X[i] * X[k]
        X_ext[pair_to_index[j, k], 0] = X[j] * X[k]
    
    X_ext.requires_grad_()
    objective = sdp_objective(X_ext, data)
    constraint = sdp_constraint(X_ext, data)
    #old code
    #objective,penalty = sat_objective(X,A,C,N,K)
    # Step 3: Choose an optimization algorithm (e.g., Stochastic Gradient Descent - SGD)
    #optimizer = Adam([X_ext], lr=0.1)#pecify the learning rate

    # Step 4: Perform the optimization loop
    num_epochs = 1000 # Number of optimization iterations
    batch_size = 50
    batch_iter = 50
    for epoch in range(num_epochs):
        # Calculate the function value and gradient
        if epoch%batch_iter == 0:
            #batch = int(epoch/100)
            #redefine X_ext
            #batch_low = int(epoch/batch_size)*batch_size
            #batch_hi = batch_low + batch_size
            #clauses_batch = clauses[batch_low:batch_hi]
            num = len(clauses)
            select = sample_unique_numbers(0,num-1,batch_size)
            clauses_batch = clauses[select]
            batch_set = set() #set of single vars
            pairs_set = set() #set of pair vars
            for cl in clauses_batch: 
                batch_set |= set(cl)
                for i in range(len(cl)):
                    pr = cl.copy()
                    pr = np.delete(cl,i)
                    pairs_set.add(tuple(pr))
            batch_rows = torch.zeros(len(batch_set) + len(pairs_set))
            index = 0
            for single in batch_set:
                batch_rows[index] = single
                index += 1
            for pair in pairs_set:
                batch_rows[index] = pair_to_index[pair[0],pair[1]]
                index += 1
            batch_rows = batch_rows.int()
            #X_batch = X_ext[batch_rows]
            #print('X batch: ', X_batch)
            # Detach and clone the slice to create a new leaf tensor
            #X_ext.detach()
            #X_batch = X_ext[batch_rows].detach().clone().requires_grad_(True)
            optimizer = Adam([X_ext], lr=0.1) #specify the learning rate
            remainder = torch.tensor(list(set([torch.tensor(i) for i in range(X_ext.shape[0])]) - set(batch_rows)))
            X_ext[remainder].detach()
            #for i in range(X_ext.shape[0]):
            #    if i not in batch_rows:
            #        X_ext[i].detach()
            #X_ext[batch_rows] = X_batch
            #fix variables associated to clauses    
        objective = sdp_objective(X_ext, data)
        constraint = sdp_constraint(X_ext, data)
        norm = sdp_norm(X_ext,data)
        #mu = 0.005
        mu = 0.01
        if epoch%100 == 0:
            print('epoch: ', epoch)
            print('objective: ', objective)
            print('constraint: ', mu*constraint)
        obj = -1.0*objective + mu*(constraint + norm)
        #print('obj: ', obj)
        # Clear the gradients from the previous iteration
        optimizer.zero_grad()
    
        # Compute the gradient of the function with respect to x
        obj.backward()
        #print('grad: ', X.grad)
        
        optimizer.step()
    
    X_ext.requires_grad_(False)
    h = 1000
    hyperplane = torch.randn(h,X_ext.shape[1],device=device)
    norms = torch.norm(hyperplane, p=2, dim=1, keepdim=True)
    hyperplane = hyperplane/norms
    ans = torch.sign(torch.matmul(X_ext,torch.transpose(hyperplane,0,1)))
    #X_ext[:,0] = torch.sign(X_ext[:,0]) 
    sat = SATProblem()
    args = None
    candidates = []
    for i in range(h):
         candidates.append(sat.score(args,ans[:,i],data))
    #print('candidates: ', candidates)
    print('sat score: ', max(candidates))
    return max(candidates)
    
    #print('sat score: ', sat.score(args,X_ext,data))    
    
def autograd(clauses,signs,X):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_vars, pair_to_index, data = compile_sat(clauses, signs, N, K)
    data = data.to(device)

    # extend random assignment into ij domain
    r = 2
    X_ext = torch.zeros(total_vars, r, device=device)
    for i in range(N):
        X_ext[i, 0] = X[i]

    for f in range(K):
        i = int(clauses[f, 0])
        j = int(clauses[f, 1])
        k = int(clauses[f, 2])
        X_ext[pair_to_index[i, j], 0] = X[i] * X[j]
        X_ext[pair_to_index[i, k], 0] = X[i] * X[k]
        X_ext[pair_to_index[j, k], 0] = X[j] * X[k]
    
    X_ext.requires_grad_()
    objective = sdp_objective(X_ext, data)
    constraint = sdp_constraint(X_ext, data)
    #old code
    #objective,penalty = sat_objective(X,A,C,N,K)
    # Step 3: Choose an optimization algorithm (e.g., Stochastic Gradient Descent - SGD)
    optimizer = Adam([X_ext], lr=0.1)#pecify the learning rate

    # Step 4: Perform the optimization loop
    num_epochs = 1000 # Number of optimization iterations
    for epoch in range(num_epochs):
        # Calculate the function value and gradient
        objective = sdp_objective(X_ext, data)
        constraint = sdp_constraint(X_ext, data)
        norm = sdp_norm(X_ext,data)
        mu = 0.005
        if epoch%100 == 0:
            print('epoch: ', epoch)
            print('objective: ', objective)
            print('constraint: ', mu*constraint)
        obj = -1.0*objective + mu*(constraint + norm)
        print('obj: ', obj)
        # Clear the gradients from the previous iteration
        optimizer.zero_grad()
    
        # Compute the gradient of the function with respect to x
        obj.backward()
        #print('grad: ', X.grad)
        
        optimizer.step()
    
    X_ext.requires_grad_(False)
    h = 1000
    hyperplane = torch.randn(h,X_ext.shape[1],device=device)
    norms = torch.norm(hyperplane, p=2, dim=1, keepdim=True)
    hyperplane = hyperplane/norms
    ans = torch.sign(torch.matmul(X_ext,torch.transpose(hyperplane,0,1)))
    #X_ext[:,0] = torch.sign(X_ext[:,0]) 
    sat = SATProblem()
    args = None
    candidates = []
    for i in range(h):
         candidates.append(sat.score(args,ans[:,i],data))
    #print('candidates: ', candidates)
    print('sat score: ', max(candidates))
    return max(candidates)
    
    #print('sat score: ', sat.score(args,X_ext,data))
    
def sdp_norm(X,data):
    squared_norms = torch.norm(X, p=2, dim=1) ** 2
    return torch.sum(torch.square(squared_norms - torch.ones(squared_norms.shape,device=device)))

def sdp_pivot(X,data,pivots):
    if pivots[0].nelement() == 0:
        return torch.tensor(0,device=device)
    piv,vals = pivots
    piv = piv.int()
    #print('X: ', X)
    #print('piv ', piv)
    fixed = torch.zeros(X[piv,:].shape, device=device)
    fixed[:,0] = vals
    return torch.sum(torch.square(torch.norm(X[piv,:] - fixed,p=2,dim=1)))

class node:
    def __init__(self,X,pivots,obj):
        self.X = X
        self.pivots = pivots
        self.obj = obj 
    
def tree_autograd(clauses,signs,X,params):
    
    #set of hyperparameters
    num_pivot = params['num_pivot']
    hyper = params['hyper']
    dimension = params['dimension']
    penalty = params['penalty']
    total_nodes = params['total_nodes']
    batch = params['batch']

    #compile sat instance 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_vars, pair_to_index, data = compile_sat(clauses, signs, N, K)
    data = data.to(device)

    # extend random assignment into ij domain
    X_ext = torch.zeros(total_vars, dimension, device=device)
    for i in range(N):
        X_ext[i, 0] = X[i]

    for f in range(K):
        i = int(clauses[f, 0])
        j = int(clauses[f, 1])
        k = int(clauses[f, 2])
        X_ext[pair_to_index[i, j], 0] = X[i] * X[j]
        X_ext[pair_to_index[i, k], 0] = X[i] * X[k]
        X_ext[pair_to_index[j, k], 0] = X[j] * X[k]
    
    #initialize tree
    #TODO: right now there is no tree just bit flipping
    tree = []
    pivots = (torch.tensor([],device=device),torch.tensor([],device=device))
    X_ext.requires_grad_()
    tree.append((X_ext,pivots,float('inf')))
    answers = []
    
    #batch handling: consider removing this
    init_epoch = 1000 #number of optimization iterations
    later_epoch = 50 #number of opt iterations after first node
    batch_size = 50 #number of frozen variables per batch
    batch_iter_init = 50 #number of iterations per batch
    batch_iter_later = 10 #number of iterations per batch after first node
    tree_limit = 100 #num_epochs * tree_limit = total iterations
    timing = [] #record of times
    #iterations of tree search 
    start_time = time.time()
    import random
    for it in range(total_nodes):
        print('tree it: ', it)
        #right now tree only has one node 
        node = tree[0]
        X_ext,pivots,score = node
        X_ext.requires_grad_()
        piv,vals = pivots
        
        #handling first node vs later nodes
        if it == 0:
            num_epochs = init_epoch # Number of optimization iterations
            batch_iter = batch_iter_init
        else:
            num_epochs = later_epoch
            batch_iter = batch_iter_later
        
        #central optimization loop
        for epoch in range(num_epochs):
            if epoch%100 == 0:
                print('epoch: ', epoch)
            # Calculate the function value and gradient
            if batch: 
                if epoch%batch_iter == 0:
                    num = len(clauses)
                    select = sample_unique_numbers(0,num-1,batch_size)
                    clauses_batch = clauses[select]
                    batch_set = set() #set of single vars
                    pairs_set = set() #set of pair vars
                    for cl in clauses_batch: 
                        batch_set |= set(cl)
                        for i in range(len(cl)):
                            pr = cl.copy()
                            pr = np.delete(cl,i)
                            pairs_set.add(tuple(pr))
                    batch_rows = torch.zeros(len(batch_set) + len(pairs_set))
                    index = 0
                    for single in batch_set:
                        batch_rows[index] = single
                        index += 1
                    for pair in pairs_set:
                        batch_rows[index] = pair_to_index[pair[0],pair[1]]
                        index += 1
                    batch_rows = batch_rows.int()
                    optimizer = Adam([X_ext], lr=0.1) #specify the learning rate
                    remainder = torch.tensor(list(set([torch.tensor(i) for i in range(X_ext.shape[0])]) - set(batch_rows)))
                    X_ext[remainder].detach()
            else: 
                 optimizer = Adam([X_ext], lr=0.1) #specify the learning rate
                
            # Calculate the function value and gradient
            objective = sdp_objective(X_ext, data)
            constraint = sdp_constraint(X_ext, data)
            norm = sdp_norm(X_ext,data)
            fixing = sdp_pivot(X_ext,data,pivots)
            obj = -1.0*objective + penalty*(constraint + norm) + 1000*fixing
            # Clear the gradients from the previous iteration
            optimizer.zero_grad()
    
            # Compute the gradient of the function with respect to x
            obj.backward()
        
            optimizer.step()
    
        X_ext.requires_grad_(False)
        hyperplane = torch.randn(hyper,X_ext.shape[1],device=device)
        norms = torch.norm(hyperplane, p=2, dim=1, keepdim=True)
        hyperplane = hyperplane/norms
        ans = torch.sign(torch.matmul(X_ext,torch.transpose(hyperplane,0,1)))
        sat = SATProblem()
        args = None
        candidates = []
        print('hyerplane rounding start')
        for i in range(hyper):
            candidates.append(sat.score(args,ans[:,i],data))
        print('hyperplane rounding end')    
        score = max(candidates)
        answers.append(score)
        print('sat score: ', score)
        print('MAX ANSWER: ', max(answers))
        #get index of max score
        #print('chunk 1')
        #index = candidates.index(max(candidates))
        #criticals = X_ext[:N,0]
        #criticals = torch.abs(criticals[:N])
        #print('min critical: ', min(criticals))
        #crit_enum = []
        #print('start for loop')
        #for i in range(N):
        #    crit_enum.append((i,criticals[i]))
        
        #print('chunk 2')
        piv = sample_unique_numbers(0, N-1, num_pivot)
        vals = 2*torch.bernoulli(torch.tensor([0.5]*num_pivot)) - torch.ones(num_pivot)
        node_x = X_ext.clone()
        vals = vals.to(device)
        node_x.to(device)
        piv = torch.tensor(piv)
        piv = piv.to(device)
        node_x[piv,0] = vals
        node_x.requires_grad_(True)
        node = (node_x,(piv,vals),score)
        
        #we would have to delete the existing node if we were working with trees
        tree = [node]
        end_time = time.time()
        timing.append(end_time - start_time)
        print('end iter')
    
    output = [max(answers[:i+1]) for i in range(len(answers))]
    return (output,timing)
    
    #print('sat score: ', sat.score(args,X_ext,data))

if __name__ == '__main__':
    print("hello! I am data/tree_autograd.py and it is time to run some tests")
    seed = 0
    N = 100
    K = 400
    
    
    params = {}
    #strong setting
    #params['num_pivot'] = 5
    #params['hyper'] = 4
    #params['dimension'] = 5
    #params['penalty'] = 0.01
    #params['total_nodes'] = 200
    
    #vanilla setting
    params['num_pivot'] = 5
    params['hyper'] = 1000
    params['dimension'] = 100
    params['penalty'] = 0.01
    params['total_nodes'] = 10
    params['batch'] = False
    
    num_runs = 20
    final_output = torch.zeros((num_runs,params['total_nodes'])) 
    final_timing = torch.zeros((num_runs,params['total_nodes']))
    
    torch.save(final_output, 'output.pt')
    torch.save(final_output, 'timing.pt')
    
    for run in range(num_runs):
        print('RUN: ', run)
        random_state = np.random.RandomState(seed)
        clauses, signs = random_3sat_clauses(random_state, N, K)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = torch.bernoulli(torch.full((N,), 0.5, device=device))*2-1
        output,timing = tree_autograd(clauses,signs,X,params)
        print('output: ', output)
        print('timing: ', timing)
        final_output[run,:] = torch.tensor(output)
        final_timing[run,:] = torch.tensor(timing)
    print('final output: ', final_output)
    print('final timing: ', final_timing)
    import matplotlib.pyplot as plt

    # Example 2D tensor 'Data'
    data = final_output
    time = final_timing
    # Calculate the average of each column
    time = time.mean(dim=0)

    # Calculate the average and standard deviation of each column in Data
    avg_data = data.mean(dim=0)
    std_data = data.std(dim=0)

    # Plotting
    plt.figure(figsize=(12, 8))
    # Plot line for averages
    plt.plot(time.numpy(), avg_data.numpy(), label='Average Data', linestyle='-', color='blue', marker='o')
    # Plot error bars
    plt.errorbar(time.numpy(), avg_data.numpy(), yerr=std_data.numpy(), fmt='o', ecolor='lightgray', elinewidth=3, capsize=0)
    # Add shaded error
    plt.fill_between(time.numpy(), avg_data.numpy() - std_data.numpy(), avg_data.numpy() + std_data.numpy(), color='lightblue', alpha=0.5)
    plt.ylim(390, 400)

    plt.xlabel('Time')
    plt.ylabel('Average Data')
    plt.title('Plot of Averages with Error Bars and Shading')
    plt.legend()
    plt.grid(True)
    plt.savefig('autograd_vanilla.png')
    plt.show()
    
    

