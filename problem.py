import torch


class PrimalLP:
    def __init__(self, c, nonnegative_mask):
        super().__init__()
        self.c = c
        self.nonnegative_mask = nonnegative_mask.bool()

    @staticmethod
    def optimality_gap(obj, true_obj):
        return torch.abs((true_obj - obj) / true_obj)

    def obj_fn(self, x):
        return x @ self.c

    def ineq_residual(self, x):
        return torch.relu(-x[:, self.nonnegative_mask])

    @staticmethod
    def eq_residual(x, A, b):
        return (A @ x.flatten() - b.flatten()).view(-1, b.shape[-1])



