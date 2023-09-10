# For the selected dataset, run a series of baseline computations and store the results.

import json
import os

import torch
from torch.utils.data import random_split

from data.loader import construct_dataset
from model.losses import max_cut_score, vertex_cover_score, max_clique_score
from model.parsing import parse_baseline_args
from utils.baselines import max_cut_sdp, max_cut_bm, max_cut_greedy, max_cut_gurobi, vertex_cover_gurobi
from utils.baselines import vertex_cover_sdp, vertex_cover_bm, vertex_cover_greedy
from utils.baselines import e1_projector, random_hyperplane_projector

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

    # rig to only run on the validation dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    indices = val_dataset.indices
    dataset = val_dataset

    lift_fns = {}
    if args.problem_type == 'max_cut':
        if args.sdp:
            lift_fns['sdp'] = max_cut_sdp
            #'bm': max_cut_bm,
        greedy_fn = max_cut_greedy
        score_fn = max_cut_score
    elif args.problem_type == 'vertex_cover':
        if args.sdp:
            lift_fns['sdp'] = vertex_cover_sdp
        greedy_fn = vertex_cover_greedy
        score_fn = vertex_cover_score
    elif args.problem_type == 'max_clique':
        if args.sdp:
            lift_fns['sdp'] = vertex_cover_sdp
        greedy_fn = vertex_cover_greedy
        score_fn = max_clique_score
    else:
        raise ValueError(f"baselines got invalid problem_type {args.problem_type}")

    project_fns = {
      'e1': e1_projector,
      'random_hyperplane': random_hyperplane_projector,
    }

    results = []
    for (i, example) in enumerate(dataset):
        if args.start_index is not None and i < args.start_index:
            continue

        if args.end_index is not None and i >= args.end_index:
            break

        # we do max clique by running VC on graph complement
        if args.problem_type == 'max_clique':
            nx_graph = to_networkx(example)
            nx_complement = nx.operators.complement(nx_graph)
            example = from_networkx(nx_complement)

        # we'll run each pair of lift method and project method
        for lift_name, lift_fn in lift_fns.items():
            # calculate lift output and save score
            x_lift = torch.FloatTensor(lift_fn(args, example))
            # NOTE: no penalty in return
            lift_score = score_fn(args, x_lift, example)
            res = {
                'index': indices[i],
                'method': lift_name,
                'type': 'lift',
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
                x_project = torch.FloatTensor(project_fn(args, x_lift, example, score_fn))
                # NOTE: no penalty in return
                project_score = score_fn(args, x_project, example)
                res = {
                    'index': indices[i],
                    'method': f"{lift_name}|{project_name}",
                    'type': 'lift_project',
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
            if args.problem_type == 'max_cut':
                x_gurobi = max_cut_gurobi(args, example)
                # TODO: there's no gurobi_penalty here anymore?
                # gurobi_penalty = None
                gurobi_score = score_fn(args, x_gurobi, example)
            elif args.problem_type == 'vertex_cover':
                set_size, x_gurobi = vertex_cover_gurobi(args, example)
                # TODO: there's no gurobi_penalty here anymore?
                # gurobi_penalty = None
                gurobi_score = score_fn(args, x_gurobi, example)
            elif args.problem_type == 'max_clique':
                # we do max clique by running VC on graph complement
                set_size, x_gurobi = vertex_cover_gurobi(args, example)
                # TODO: there's no gurobi_penalty here anymore?
                # gurobi_penalty = None
                gurobi_score = score_fn(args, x_gurobi, example)
            else:
                raise ValueError(f"baselines got invalid problem_type {args.problem_type}")
            print(f"Gurobi integral score {gurobi_score}")

            res = {
                'index': indices[i],
                'method': 'gurobi',
                'type': 'solver',
                'score': float(gurobi_score),
                #'penalty': float(gurobi_penalty),
                'x': x_gurobi.tolist(),
            }
            outfile.write(json.dumps(res) + '\n')
            outfile.flush()
            results.append(res)

    # TODO print some summary statistics
