# Several baseline implementations for each problem

import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_torch_csr_tensor, to_networkx
import networkx as nx


def max_cut_sdp(args, example):
    N = example.num_nodes
    edge_index = example.edge_index
    E = edge_index.shape[1]
    A = to_dense_adj(edge_index, max_num_nodes=N)[0]

    edge_weights = torch.ones(E)

    C = []
    b = []
    for i in range(N):
        const = np.zeros((N, N))
        const[i,i] = 1
        C.append(const)
        b.append(1)

    # Define and solve the CVXPY problem.
    # Create a symmetric matrix variable.
    X = cp.Variable((N, N), symmetric=True)

    # The operator >> denotes matrix inequality.
    constraints = [X >> 0]
    constraints += [
        cp.trace(C[i] @ X) == b[i] for i in range(N)
    ]
    prob = cp.Problem(cp.Minimize(cp.trace(A @ X)),
                    constraints)
    prob.solve(verbose=True)

    # Print result.
    sol = prob.value / 2.
    obj_relax = (E - sol) / 2.

    #print('SDP objective:', prob.value)
    #print("SDP solution X:",  X.value)
    #print("relaxed cut value:", obj_relax)

    X = X.value
    (eigenval, eigenvec) = np.linalg.eig(X)
    #print('Eigenvalues--Note how sparse they are:', eigenval)

    # ensure eigenvalues are positive
    # pad by .001 for precision issues with cholesky decomposition
    if np.min(eigenval) < 0:
        X = X + (0.001 - np.min(eigenval)) * np.eye(N)
    V = np.linalg.cholesky(X)
    #print('Cholesky Decomposition:', V)

    return V

def max_cut_bm():
    pass # TODO

def max_cut_autograd():
    pass # TODO

def vertex_cover_sdp(args, example):
    # from /maxcut-80/vertex_cover/vc_sdp.ipynb::vc_sdp
    # Define the variable representing the relaxed vertex cover solution
    N = example.num_nodes
    edge_index = example.edge_index
    A = to_dense_adj(edge_index, max_num_nodes=N)[0]


    X = cp.Variable((N+1, N+1), PSD=True)
    # TODO weights?
    weight_mat = np.zeros((N+1,N+1))
    weight_mat[0,1:N+1] = 1

    # Objective function (minimize trace of X)
    objective = cp.Minimize(cp.sum(cp.multiply(weight_mat,X)))

    # Constraints: X is a positive semidefinite matrix
    constraints = [X >> 0]

    # Constraints: diagonal elements of X are 1
    constraints += [cp.diag(X) == 1]

    # Constraints: for each edge (i, j) one must be nonzero, note the indexing is (i,j) in A
    constraints += [1 - X[0, i+1] - X[0,j+1] + X[i+1,j+1] == 0 for i in range(N) for j in range(N) if A[i, j] == 1]

    # Create the problem and solve it
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=True)

    X = X.value

    # NOTE: removed calculations for int_obj, obj, frac_obj, etc
    
    eigenval, _  = np.linalg.eig(X)
    # ensure eigenvalues are positive
    # pad by .001 for precision issues with cholesky decomposition
    if np.min(eigenval) < 0:
        X = X + (0.001 - np.min(eigenval)) * np.eye(N+1)
    V = np.linalg.cholesky(X)

    V = V[1:N+1,:]

    return V

def vertex_cover_bm():
    pass # TODO

def vertex_cover_autograd():
    pass # TODO

def e1_projector(args, x_lift, example, score_fn):
    if isinstance(x_lift, np.ndarray):
        x_lift = torch.FloatTensor(x_lift)
    return torch.sign(x_lift[:, 0, None])

def random_hyperplane_projector(args, x_lift, example, score_fn):
    if isinstance(x_lift, np.ndarray):
        x_lift = torch.FloatTensor(x_lift)
    N = example.num_nodes
    edge_index = example.edge_index.to(x_lift.device)
    E = edge_index.shape[1]
    A = to_dense_adj(edge_index)[0]
    A = to_dense_adj(edge_index, max_num_nodes=N)[0]

    outputs = []
    n_hyperplanes = 1000 # TODO make this modifiable in args

    hyper = torch.randn((n_hyperplanes, x_lift.shape[1]), device=x_lift.device)
    hyper = F.normalize(hyper)
    x_proj = torch.matmul(hyper, x_lift.t())
    x_int = torch.sign(x_proj)[:, :, None]
    scores = torch.vmap(lambda x: score_fn(args, x, example))(x_int)
    best = torch.argmax(scores)
    out = x_int[best, :, 0]
    return out

def max_cut_greedy(args, x_proj, example, score_fn):
    pass # TODO

def vertex_cover_greedy(A, warm_start=None, iterations=100, weights=None):
    # from /maxcut-80/vertex_cover/vc_sdp.ipynb::vc_greedy
    #start = np.random.bernoulli(N)
    N , _ = A.shape
    if weights is None:
        weights = np.ones(N)
        
    if warm_start is None:
        start = np.ones(N)
    else: 
        start = warm_start
    for it in range(iterations):
        for i in range(N):
            all_ones = True
            for j in range(N):
                if A[i,j] == 0:
                    continue
                if A[i,j] == 1 and start[j] == 0:
                    all_ones = False
            if all_ones and weights[i] > 0:
                start[i] = 0
            if not all_ones and start[i] == 0:
                start[i] = 1
    return np.inner(weights,start)

def max_cut_gurobi(args, example):
    # Create a new model
    m = gp.Model("maxcut")
    m.params.OutputFlag = 0

    # time limit in seconds, if applicable
    if args.gurobi_timeout:
        m.params.TimeLimit = args.gurobi_timeout

    # Set up node variables
    x_vars = {}
    for i in range(example.num_nodes):
        x_vars["x_" + str(i)] = m.addVar(vtype=GRB.BINARY, name="x_" + str(i))

    r,c = example.edge_index
    # Set objective
    obj = gp.QuadExpr()
    #Iterate over edges to compute (x_i - x_j)**2 for each edge (i,j) and sum it all up
    for source, target in zip(r,c):
        qi_qj = (x_vars['x_' + str(source.item())] - x_vars['x_' + str(target.item())])
        obj += qi_qj * qi_qj / 2
    m.setObjective(obj, GRB.MAXIMIZE)

    # Optimize model
    m.optimize()

    print("model status:", m.status)

    set_size = m.objVal
    x_vals = None
    try:
        x_vals = np.array([var.X for var in m.getVars()]) * 2 - 1
    except:
        try:
            print("Issue using var.X; trying var.x")
            x_vals = np.array([var.x for var in m.getVars()]) * 2 - 1
        except:
            print("didn't work either?!?! retrying!!!")
            return max_cut_gurobi(args, example)

    return x_vals

def vertex_cover_gurobi(args, example):
    nx_complement = to_networkx(example) # nx.operators.complement()
    x_vars = {}
    m = gp.Model("mip1")
    m.params.OutputFlag=0

    if args.gurobi_timeout:
        m.params.TimeLimit = args.gurobi_timeout

    for node in nx_complement.nodes():
        x_vars['x_'+str(node)] = m.addVar(vtype=GRB.BINARY, name="x_"+str(node))

    count_edges = 0
    for edge in nx_complement.edges():
        m.addConstr(x_vars['x_'+str(edge[0])] + x_vars['x_'+str(edge[1])] >= 1,'c_'+str(count_edges))
        count_edges+=1
    m.setObjective(sum([x_vars['x_'+str(node)] for node in nx_complement.nodes()]), GRB.MINIMIZE);

    # Optimize model
    m.optimize();

    set_size = m.objVal;
    x_vals = np.array([var.x for var in m.getVars()]) * 2 - 1

    return set_size, x_vals
