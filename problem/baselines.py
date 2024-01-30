# Several baseline implementations for each problem

import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_torch_csr_tensor, to_networkx
from torch_geometric.data import Batch
import networkx as nx
import mosek

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

    return V, prob.status, prob.solver_stats.solve_time

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
    problem.solve(solver=cp.MOSEK, mosek_params={mosek.dparam.optimizer_max_time: 10})

    X = X.value

    # NOTE: removed calculations for int_obj, obj, frac_obj, etc
    
    eigenval, _  = np.linalg.eig(X)
    # ensure eigenvalues are positive
    # pad by .001 for precision issues with cholesky decomposition
    if np.min(eigenval) < 0:
        X = X + (0.001 - np.min(eigenval)) * np.eye(N+1)
    V = np.linalg.cholesky(X)

    V = V[1:N+1,:]

    return V, problem.status, problem.solver_stats.solve_time

def vertex_cover_bm():
    pass # TODO

def vertex_cover_autograd():
    pass # TODO

# returns a torch.FloatTensor size (N, 1)
def e1_projector(args, x_lift, example, score_fn):
    if isinstance(x_lift, np.ndarray):
        x_lift = torch.FloatTensor(x_lift)
    return torch.sign(x_lift[:, 0, None])

# returns a torch.FloatTensor size (N,)
# n_hyperplanes: how many to try?
# n_groups: we may do it in groups to reduce memory consumption; how many?
def random_hyperplane_projector(args, x_lift, batch, score_fn, n_hyperplanes=1000, n_groups=1):
    if isinstance(x_lift, np.ndarray):
        x_lift = torch.FloatTensor(x_lift)

    #torch.cuda.memory._record_memory_history(enabled=True, trace_alloc_max_entries=100000, trace_alloc_record_context=True)
    x_int = []
    scores = []
    for i in range(n_groups):
        hyper = torch.randn((n_hyperplanes // n_groups, x_lift.shape[1]), device=x_lift.device)
        hyper = F.normalize(hyper)
        x_proj = torch.matmul(hyper, x_lift.t())

        # group_x_int[i, j] is the assignment for hyperplane i, variable j
        group_x_int = torch.sign(x_proj)[:, :, None]
        x_int.append(group_x_int)

        group_scores = []

        if isinstance(batch, Batch):
            # here we score on each component of the batch individually
            for i, example in enumerate(batch.to_data_list()):
                # slice out this example's nodes
                example_x_int = group_x_int[:, batch.ptr[i]:batch.ptr[i + 1], :]

                # XXX the penalty hack here is not ok under many conditions!
                example.penalty = batch.penalty

                # compute scores for _all_ hyperplanes and _only_ this example
                group_example_scores = torch.vmap(lambda x: score_fn(args, x, example))(example_x_int)
                group_scores.append(group_example_scores)
        else:
            # we are compatible with single-example hyperplane rounding
            group_example_scores = torch.vmap(lambda x: score_fn(args, x, batch))(group_x_int)
            group_scores.append(group_example_scores)

        # stack these scores to form a (hyperplanes x graphs) tensor
        group_scores = torch.stack(group_scores, dim=1)
        scores.append(group_scores)

    x_int = torch.cat(x_int, dim=0) # now (n_hyperplanes x nodes_in_batch x 1)
    scores = torch.cat(scores, dim=0) # now (n_hyperplanes x graphs_in_batch)

    best = torch.argmax(scores, dim=0) # now (graphs_in_batch), best hyperplane index for each graph

    out = []
    if isinstance(batch, Batch):
        for i in range(len(batch)):
            out.append(x_int[best[i], batch.ptr[i]:batch.ptr[i + 1], 0])
    else:
        out.append(x_int[best[0], :, 0])
    out = torch.cat(out, dim=0)

    num_zeros = (out == 0).count_nonzero()
    if num_zeros > 0:
        print("WARNING: detected zeros in hyperplane rounding output")

    #import pickle
    #pickle.dump(torch.cuda.memory._snapshot(), open('snapshot.pickle', 'wb'))
    #torch.cuda.memory._record_memory_history(enabled=None)
    #exit(0)

    return out

# expect a (N,) shaped x_proj, all +/- 1. will tolerate 0 entries
def generic_greedy(args, x_proj, example, score_fn, batch_sz=64, iterations=1000):
    if isinstance(x_proj, np.ndarray):
        x_proj = torch.FloatTensor(x_proj)

    N = x_proj.shape[0]

    # TODO make number of iterations adjustable from arguments
    flip_matrix = torch.ones(N, N, device=args.device) - 2 * torch.diag(torch.ones(N, device=args.device))
    current_score = score_fn(args, x_proj, example)
    for i in range(iterations):
        # spam out N versions
        versions = x_proj.repeat(N, 1) # (N, N) where each row versions[i] is a copy of x_proj

        # flip the elements on the diagonal, generating N new versions
        versions = versions * flip_matrix

        # get a score for each row and select the best version
        scores = []
        for batch_idx in range(0, N, batch_sz):
            version_slice = versions[batch_idx : min(batch_idx + batch_sz, N)]
            batch_scores = torch.vmap(lambda x: score_fn(args, x, example))(version_slice)
            scores.append(batch_scores)
        scores = torch.cat(scores)
        best = torch.argmax(scores)
        if best > current_score:
            # set the new current x_proj
            current_score = best
            x_proj = versions[best, :]
        else:
            # no version was better
            print(f"greedy terminated at {i} iterations")
            break
    else:
        print("greedy did not terminate")

    return x_proj

def max_cut_gurobi(args, example):
    # Create a new model
    m = gp.Model("maxcut")
    m.params.OutputFlag = 0
    m.params.Threads = 8
    m.setParam(GRB.Param.Threads, 8)

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

    return x_vals, m.status, m.Runtime

def vertex_cover_gurobi(args, example):
    nx_complement = to_networkx(example) # nx.operators.complement()
    x_vars = {}
    m = gp.Model("mip1")
    m.params.OutputFlag=0
    m.params.Threads = 8
    m.setParam(GRB.Param.Threads, 8)

    if args.gurobi_timeout:
        m.params.TimeLimit = args.gurobi_timeout

    for node in nx_complement.nodes():
        x_vars['x_'+str(node)] = m.addVar(vtype=GRB.BINARY, name="x_"+str(node))

    count_edges = 0
    for edge in nx_complement.edges():
        m.addConstr(x_vars['x_'+str(edge[0])] + x_vars['x_'+str(edge[1])] >= 1,'c_'+str(count_edges))
        count_edges+=1
    m.setObjective(sum([x_vars['x_'+str(node)] for node in nx_complement.nodes()]), GRB.MINIMIZE)

    # Optimize model
    m.optimize()

    x_vals = None
    try:
        x_vals = np.array([var.X for var in m.getVars()]) * 2 - 1
    except:
        try:
            print("Issue using var.X; trying var.x")
            x_vals = np.array([var.x for var in m.getVars()]) * 2 - 1
        except:
            print("didn't work either?!?! retrying!!!")
            return vertex_cover_gurobi(args, example)

    return x_vals, m.status, m.Runtime
