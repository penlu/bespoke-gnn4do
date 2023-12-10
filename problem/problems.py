from problem.losses import max_cut_obj, vertex_cover_obj, vertex_cover_constraint
from problem.losses import max_cut_score, vertex_cover_score, max_clique_score
from networkx.algorithms.approximation import one_exchange, min_weighted_vertex_cover
from problem.baselines import max_cut_sdp, vertex_cover_sdp
from problem.baselines import max_cut_gurobi, vertex_cover_gurobi
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
        X = torch.cat([torch.zeros(1, X.shape[1], device=X.device), X], dim=0)
        X[0, 0] = 1.

        # calculate objective
        XX = torch.matmul(X, torch.transpose(X, 0, 1))
        return torch.sparse.sum(batch.A * XX)

    @staticmethod
    def constraint(X, batch):
        X = torch.cat([torch.zeros(1, X.shape[1], device=X.device), X], dim=0)
        X[0, 0] = 1.

        # calculate objective
        XX = torch.matmul(X, torch.transpose(X, 0, 1))
        penalties = torch.sparse.sum(batch.C * XX, dim=(1, 2)).to_dense()
        return torch.sum(penalties * penalties)

    @staticmethod
    def loss(X, batch):
        X = torch.cat([torch.zeros(1, X.shape[1], device=X.device), X], dim=0)
        X[0, 0] = 1.

        # calculate objective
        XX = torch.matmul(X, torch.transpose(X, 0, 1))
        objective = torch.trace(torch.matmul(batch.A, XX))

        # calculate penalties
        x1_i = X[batch.C[:, 0]]
        x1_j = X[batch.C[:, 1]]
        x2_i = X[batch.C[:, 2]]
        x2_j = X[batch.C[:, 3]]

        X1 = torch.sum(x1_i * x1_j, dim=1)
        X2 = torch.sum(x2_i * x2_j, dim=1)

        penalties = X2 - X1

        return -objective + batch.penalty * torch.sum(penalties * penalties)

    @staticmethod
    def score(args, X, example):
        from pdb import set_trace as bp
        #bp()
        # python train.py --problem_type sat --dataset random-sat --batch_size 1 --infinite True --valid_freq 100 --stepwise True
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        if len(X.shape) == 1:
            X = X[:, None]
        X = torch.cat([torch.zeros(1, X.shape[1], device=X.device), X], dim=0)
        X[0, 0] = 1.

        # calculate objective
        XX = torch.matmul(X, torch.transpose(X, 0, 1))
        objective = torch.trace(torch.matmul(example.A, XX))

        x1_i = X[example.C[:, 0]]
        x1_j = X[example.C[:, 1]]
        x2_i = X[example.C[:, 2]]
        x2_j = X[example.C[:, 3]]

        X1 = torch.sum(x1_i * x1_j, dim=1)
        X2 = torch.sum(x2_i * x2_j, dim=1)

        penalties = X2 - X1

        return objective - example.penalty * torch.sum(penalties * penalties)

    @staticmethod
    def greedy(G):
        pass

    @staticmethod
    def sdp(args, example):
        pass

    @staticmethod
    def gurobi(args, example):
        pass
