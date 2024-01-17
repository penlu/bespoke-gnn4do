from problem.losses import max_cut_obj, vertex_cover_obj, vertex_cover_constraint
from problem.losses import max_cut_score, vertex_cover_score, max_clique_score
from networkx.algorithms.approximation import one_exchange, min_weighted_vertex_cover
from problem.baselines import max_cut_sdp, vertex_cover_sdp
from problem.baselines import max_cut_gurobi, vertex_cover_gurobi
from data.sat import sdp_objective, sdp_constraint
import torch
import numpy as np

def get_problem(args):
    if args.problem_type == 'max_cut':
        return MaxCutProblem
    elif args.problem_type == 'vertex_cover':
        return VertexCoverProblem
    elif args.problem_type == 'max_clique':
        return MaxCliqueProblem
    elif args.problem_type == 'sat':
        return SATProblem
    else:
        raise ValueError(f"get_problem got invalid problem_type {args.problem_type}")

# Bundle losses, constraints, and utilities for a constrained optimization problem
class OptProblem():
    @staticmethod
    def loss(X, batch):
        raise NotImplementedError()

    @staticmethod
    def objective(X, batch):
        raise NotImplementedError()

    @staticmethod
    def constraint(X, batch):
        raise NotImplementedError()

    @staticmethod
    def loss(X, batch):
        raise NotImplementedError()

    @staticmethod
    def score(args, X, example):
        raise NotImplementedError()

    @staticmethod
    def greedy(G):
        raise NotImplementedError()

    @staticmethod
    def sdp(args, example):
        raise NotImplementedError()

    @staticmethod
    def gurobi(args, example):
        raise NotImplementedError()

class MaxCutProblem(OptProblem):
    @staticmethod
    def objective(X, batch):
        return max_cut_obj(X, batch)

    @staticmethod
    def constraint(X, batch):
        return 0.

    @staticmethod
    def loss(X, batch):
        return max_cut_obj(X, batch)

    @staticmethod
    def score(args, X, example):
        return max_cut_score(args, X, example)

    @staticmethod
    def greedy(G):
        greedy_score, _ = one_exchange(G)
        return greedy_score

    @staticmethod
    def sdp(args, example):
        return max_cut_sdp(args, example)

    @staticmethod
    def gurobi(args, example):
        return max_cut_gurobi(args, example)

class VertexCoverProblem(OptProblem):
    @staticmethod
    def objective(X, batch):
        return vertex_cover_obj(X, batch)

    @staticmethod
    def constraint(X, batch):
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        if len(X.shape) == 1:
            X = X[:, None]

        return vertex_cover_constraint(X, batch)

    @staticmethod
    def loss(X, batch):
        return vertex_cover_obj(X, batch) + \
            batch.penalty * vertex_cover_constraint(X, batch)

    @staticmethod
    def score(args, X, example):
        return vertex_cover_score(args, X, example)

    @staticmethod
    def greedy(G):
        cover = min_weighted_vertex_cover(G)
        greedy_score = -len(cover)
        return greedy_score

    @staticmethod
    def sdp(args, example):
        return vertex_cover_sdp(args, example)

    @staticmethod
    def gurobi(args, example):
        return vertex_cover_gurobi(args, example)

class SATProblem(OptProblem):
    @staticmethod
    def objective(X, batch):
        return -sdp_objective(X, batch)

    @staticmethod
    def constraint(X, batch):
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        if len(X.shape) == 1:
            X = X[:, None]

        pair_index = batch.pair_index
        X[pair_index[0]] = X[pair_index[1]] * X[pair_index[2]]

        return sdp_constraint(X, batch)

    @staticmethod
    def loss(X, batch):
        objective = sdp_objective(X, batch)
        constraint = sdp_constraint(X, batch)

        return -objective + batch.penalty * constraint

    @staticmethod
    def score(args, X, example):
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        if len(X.shape) == 1:
            X = X[:, None]

        # recompute pair variables from singles, to avoid constraint violation
        pair_index = example.pair_index
        X[pair_index[0]] = X[pair_index[1]] * X[pair_index[2]]

        # TODO need to sanity check this objective -- reports better than 7/8 even without training
        # compare assigned score vs. actual SAT clauses

        objective = sdp_objective(X, example)
        constraint = sdp_constraint(X, example)

        return objective - example.penalty * constraint

    @staticmethod
    def greedy(G):
        pass

    @staticmethod
    def sdp(args, example):
        pass

    @staticmethod
    def gurobi(args, example):
        pass
