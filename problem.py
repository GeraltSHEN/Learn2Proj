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



