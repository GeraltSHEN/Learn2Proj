import torch


class PrimalLP:
    def __init__(self, c, nonnegative_mask):
        super().__init__(c)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.c = c.to(self.device)
        self.nonnegative_mask = nonnegative_mask.to(self.device)

    @staticmethod
    def optimality_gap(obj, true_obj):
        return (true_obj - obj) / true_obj

    def obj_fn(self, x):
        return x @ self.c



