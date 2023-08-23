# Several baseline implementations for each problem

import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_torch_csr_tensor

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

def vertex_cover_sdp():
    pass # TODO

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
    x_lift = x_lift.to(torch.device("cpu"))
    N = example.num_nodes
    edge_index = example.edge_index.to(torch.device("cpu"))
    E = edge_index.shape[1]
    A = to_dense_adj(edge_index)[0]
    A = to_dense_adj(edge_index, max_num_nodes=N)[0]

    outputs = []
    for i in range(1000): 
        # pick a random vector
        hyper = torch.randn(x_lift.shape[1], device=torch.device("cpu"))
        hyper = F.normalize(hyper, dim=0)

        # project onto vector
        x_proj = torch.matmul(x_lift, hyper)
        x_int = torch.sign(x_proj)[:, None]

        outputs.append((score_fn(args, x_int, example), x_int))

    outputs.sort(reverse=True, key=lambda x: x[0])
    return outputs[0][1]

def max_cut_greedy(args, x_proj, example, score_fn):
    pass # TODO

def vertex_cover_greedy(args, x_proj, example, score_fn):
    pass # TODO

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

    set_size = m.objVal
    x_vals = np.array([var.x for var in m.getVars()])[:, None] * 2 - 1

    return x_vals
