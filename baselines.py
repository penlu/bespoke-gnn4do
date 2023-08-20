# For the selected dataset, run a series of baseline computations and store the results.

import json
import numpy as np
import os

import cvxpy as cp

import torch
from torch_geometric.utils import to_dense_adj, to_torch_csr_tensor

import gurobipy as gp
from gurobipy import GRB

from model.parsing import parse_baseline_args
from model.training import featurize_batch
from data.loader import construct_dataset

def max_cut_sdp(args, example):
    N = example.num_nodes
    edge_index = example.edge_index
    E = edge_index.shape[1]
    A = to_dense_adj(edge_index)[0]

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
    prob.solve()

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
        X = X + (0.0001 - np.min(eigenval)) * np.eye(N)
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
    N = example.num_nodes
    edge_index = example.edge_index
    E = edge_index.shape[1]
    A = to_dense_adj(edge_index)[0]

    outputs = []
    for i in range(1000): 
        # pick a random vector
        hyper = np.random.randn(x_lift.shape[1])
        hyper = hyper / np.linalg.norm(hyper)

        # project onto vector
        x_proj = torch.matmul(x_lift, torch.from_numpy(hyper.astype(np.float32)))
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

def max_cut_score(args, X, example):
    # convert numpy array to torch tensor
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X)
    edge_index = example.edge_index
    A = to_torch_csr_tensor(edge_index)
    E = edge_index.shape[0]
    XX = torch.matmul(X, torch.transpose(X, 0, 1))
    obj = torch.trace(torch.matmul(A, XX)) / 2.

    return (E - obj) / 2., 0.

def vertex_cover_score():
    pass # TODO

if __name__ == '__main__':
    # parse args
    args = parse_baseline_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)

    # save params
    args.device = str(args.device)
    json.dump(vars(args), open(os.path.join(args.log_dir, 'params.txt'), 'w'))
    outfile = open(os.path.join(args.log_dir, 'results.jsonl'), 'w')

    # get data
    dataset = construct_dataset(args)

    if args.problem_type == 'max_cut':
        lift_fns = {
          'sdp': max_cut_sdp,
          #'bm': max_cut_bm,
        }
        greedy_fn = max_cut_greedy
        score_fn = max_cut_score
    elif args.problem_type == 'vertex_cover':
        lift_fns = {
          'sdp': vertex_cover_sdp,
          'bm': vertex_cover_bm,
        }
        greedy_fn = vertex_cover_greedy
        score_fn = vertex_cover_score
    elif args.problem_type == 'max_clique':
        raise NotImplementedError(f"max_clique baselines not yet implemented")
    else:
        raise ValueError(f"baselines got invalid problem_type {args.problem_type}")

    project_fns = {
      'e1': e1_projector,
      'random_hyperplane': random_hyperplane_projector,
    }

    results = []
    for (i, example) in enumerate(dataset):
        # we'll run each pair of lift method and project method
        for lift_name, lift_fn in lift_fns.items():
            # calculate lift output and save score
            x_lift = lift_fn(args, example)
            lift_score, lift_penalty = score_fn(args, x_lift, example)
            res = {
                'index': i,
                'method': lift_name,
                'type': 'lift',
                'score': float(lift_score),
                'penalty': float(lift_penalty),
                'x': x_lift.tolist(),
            }
            outfile.write(json.dumps(res) + '\n')
            results.append(res)
            print(f"Lift method {lift_name} fractional score {lift_score}")

            # now use each project method and save scores
            for project_name, project_fn in project_fns.items():
                x_project = project_fn(args, x_lift, example, score_fn)
                project_score, project_penalty = score_fn(args, x_project, example)
                res = {
                    'index': i,
                    'method': f"{lift_name}|{project_name}",
                    'type': 'lift_project',
                    'score': float(project_score),
                    'penalty': float(project_penalty),
                    'x': x_project.tolist(),
                }
                outfile.write(json.dumps(res) + '\n')
                results.append(res)
                print(f"  Project method {project_name} integral score {project_score}")

        # run gurobi
        if args.gurobi:
            if args.problem_type == 'max_cut':
                x_gurobi = max_cut_gurobi(args, example)
                gurobi_score, gurobi_penalty = score_fn(args, x_gurobi, example)
            elif args.problem_type == 'vertex_cover':
                x_gurobi = vertex_cover_gurobi(args, example)
                gurobi_score, gurobi_penalty = score_fn(args, x_gurobi, example)
            elif args.problem_type == 'max_clique':
                raise NotImplementedError(f"max_clique baselines not yet implemented")
            else:
                raise ValueError(f"baselines got invalid problem_type {args.problem_type}")
            print(f"Gurobi integral score {gurobi_score}")

            res = {
                'index': i,
                'method': 'gurobi',
                'type': 'solver',
                'score': float(gurobi_score),
                'penalty': float(gurobi_penalty),
                'x': x_gurobi.tolist(),
            }
            outfile.write(json.dumps(res) + '\n')
            results.append(res)

    # TODO print some summary statistics
