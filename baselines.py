# For the selected dataset, run a series of baseline computations and store the results.

import itertools
import json
import os
import time
import traceback

import torch
from torch.utils.data import IterableDataset, random_split
from torch_geometric.utils.convert import to_networkx

from networkx.algorithms.approximation import one_exchange, min_weighted_vertex_cover

from data.loader import construct_dataset
from model.losses import max_cut_score, vertex_cover_score, max_clique_score
from model.parsing import parse_baseline_args
from utils.baselines import max_cut_sdp, max_cut_bm, max_cut_gurobi, vertex_cover_gurobi
from utils.baselines import generic_greedy
from utils.baselines import e1_projector, random_hyperplane_projector
from utils.graph_utils import complement_graph

if __name__ == '__main__':
    # parse args
    args = parse_baseline_args()
    print(args)
    torch.manual_seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)

    # save params
    args.device = str(args.device)
    json.dump(vars(args), open(os.path.join(args.log_dir, 'params.txt'), 'w'))
    outfile = open(os.path.join(args.log_dir, 'results.jsonl'), 'w')

    # get data
    dataset = construct_dataset(args)
    if isinstance(dataset, IterableDataset):
        val_set = list(itertools.islice(dataset, 1000))
        test_set = list(itertools.islice(dataset, 1000))
    else:
        train_size = int(0.8 * len(dataset))
        val_size = (len(dataset) - train_size)//2
        test_size = len(dataset) - train_size - val_size

        generator = torch.Generator().manual_seed(args.split_seed)
        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    lift_fns = {}
    if args.problem_type == 'max_cut':
        if args.sdp:
            lift_fns['sdp'] = max_cut_sdp
            #'bm': max_cut_bm,
        score_fn = max_cut_score
    elif args.problem_type == 'vertex_cover':
        if args.sdp:
            lift_fns['sdp'] = vertex_cover_sdp
        score_fn = vertex_cover_score
    elif args.problem_type == 'max_clique':
        if args.sdp:
            lift_fns['sdp'] = vertex_cover_sdp
        score_fn = max_clique_score
    else:
        raise ValueError(f"baselines got invalid problem_type {args.problem_type}")

    project_fns = {
      'e1': e1_projector,
      'random_hyperplane': random_hyperplane_projector,
    }

    results = []
    for (i, example) in enumerate(test_set):
        if args.start_index is not None and i < args.start_index:
            continue

        if args.end_index is not None and i >= args.end_index:
            break

        # we'll run each pair of lift method and project method
        for lift_name, lift_fn in lift_fns.items():
            # calculate lift output and save score
            start_time = time.time()
            try:
                x_lift, status, runtime = lift_fn(args, example)
                x_lift = torch.FloatTensor(x_lift)
            except:
                # if it blows up, we just ignore this example and move on.
                # blowing up is probably due to MOSEK timeout
                traceback.print_exc()
                print(f"Caught exception at example {i} method {lift_name}; skipping.")
                continue
            lift_time = time.time() - start_time

            # NOTE: no penalty in return
            lift_score = score_fn(args, x_lift, example)
            res = {
                'index': i,
                'method': lift_name,
                'type': 'lift',
                'status': status,
                'time': lift_time,
                'runtime': runtime, # as reported by problem.solver_stats.solve_time
                'score': float(lift_score),
                #'penalty': float(lift_penalty),
                'x': x_lift.tolist(),
            }
            outfile.write(json.dumps(res) + '\n')
            outfile.flush()
            results.append(res)
            print(f"Lift method {lift_name} fractional score {lift_score}")

            # now use each project method and save scores
            for project_name, project_fn in project_fns.items():
                start_time = time.time()
                x_project = torch.FloatTensor(project_fn(args, x_lift, example, score_fn))
                proj_time = time.time() - start_time

                # NOTE: no penalty in return
                project_score = score_fn(args, x_project, example)
                res = {
                    'index': i,
                    'method': f"{lift_name}|{project_name}",
                    'type': 'lift_project',
                    'time': proj_time,
                    'score': float(project_score),
                    #'penalty': float(project_penalty),
                    'x': x_project.tolist(),
                }
                outfile.write(json.dumps(res) + '\n')
                outfile.flush()
                results.append(res)
                print(f"  Project method {project_name} integral score {project_score}")

        # run gurobi
        if args.gurobi:
            start_time = time.time()
            if args.problem_type == 'max_cut':
                x_gurobi, status, runtime = max_cut_gurobi(args, example)
            elif args.problem_type == 'vertex_cover':
                x_gurobi, status, runtime = vertex_cover_gurobi(args, example)
            elif args.problem_type == 'max_clique':
                # we do max clique by running VC on graph complement
                x_gurobi, status, runtime = vertex_cover_gurobi(args, example)
            else:
                raise ValueError(f"baselines got invalid problem_type {args.problem_type}")
            gurobi_time = time.time() - start_time

            gurobi_score = score_fn(args, x_gurobi, example)
            print(f"Gurobi integral score {gurobi_score}")

            res = {
                'index': i,
                'method': 'gurobi',
                'type': 'solver',
                'status': status,
                'time': gurobi_time,
                'runtime': runtime, # as reported by m.Runtime
                'score': float(gurobi_score),
                #'penalty': float(gurobi_penalty),
                'x': x_gurobi.tolist(),
            }
            outfile.write(json.dumps(res) + '\n')
            outfile.flush()
            results.append(res)

        # run greedy
        if args.greedy:
            G = to_networkx(example, to_undirected=True)
            start_time = time.time()
            if args.problem_type == 'max_cut':
                greedy_score, _ = one_exchange(G)
            elif args.problem_type == 'vertex_cover':
                cover = min_weighted_vertex_cover(G)
                greedy_score = -len(cover)
            else:
                raise ValueError(f"greedy baseline not implemented for problem {args.problem_type}")
            greedy_time = time.time() - start_time

            res = {
                'index': i,
                'method': 'greedy',
                'type': 'solver',
                'time': greedy_time,
                'score': float(greedy_score),
            }
            outfile.write(json.dumps(res) + '\n')
            outfile.flush()
            results.append(res)

    # TODO print some summary statistics
