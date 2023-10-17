from problem.losses import max_cut_loss, vertex_cover_loss, max_clique_loss
from problem.losses import max_cut_score, vertex_cover_score, max_clique_score

def get_problem(args):
    if args.problem_type == 'max_cut':
        return MaxCutProblem
    elif args.problem_type == 'vertex_cover':
        return VertexCoverProblem
    elif args.problem_type == 'max_clique':
        return MaxCliqueProblem

# Bundle losses, constraints, and utilities for a constrained optimization problem
class OptProblem():
    @staticmethod
    @abstractmethod
    def objective(X, batch):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def constraint(X, batch):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def score(args, X, example):
        raise NotImplementedError()

class MaxCutProblem(OptProblem):
    @staticmethod
    def objective(X, batch):
        return max_cut_obj(X, batch)

    @staticmethod
    def constraint(X, batch):
        return 0.

    @staticmethod
    def score(args, X, example):
        return max_cut_score(args, X, example)

class VertexCoverProblem(OptProblem):
    @staticmethod
    def objective(X, batch):
        return vertex_cover_obj(X, batch)

    @staticmethod
    def constraint(X, batch):
        return vertex_cover_constraint(X, batch)

    @staticmethod
    def score(args, X, example):
        return vertex_cover_score(args, X, example)
