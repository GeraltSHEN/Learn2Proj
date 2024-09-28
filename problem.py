import torch


class PrimalLP:
    """
        minimize_x c^T x
        s.t.       Ax = b
                   s >= 0
    where b will be inputs and x will be outputs in the neural network
    """
    def __init__(self, c, A, free_idx, truncate_idx):
        self.name = 'primal_lp'
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._c = c
        self._A = A

        self._const_num = A.shape[0]
        self._var_num = A.shape[1]
        self._free_idx = free_idx  # free_idx[0] = 0, free_idx[1] = free_num - 1, made by default
        self._free_num = free_idx[1] + 1
        self._truncate_idx = truncate_idx

        self._free_W = torch.eye(A.shape[1])[free_idx[0]:free_idx[1]].to(self._device)
        self._s_W = torch.cat((torch.eye(A.shape[1])[:free_idx[0]], torch.eye(A.shape[1])[free_idx[1]+1:]), dim=0).to(self._device)

        self._mutable_W = torch.eye(A.shape[1])[truncate_idx[0]:truncate_idx[1]].to(self._device)
        self._immutable_W = torch.cat((torch.eye(A.shape[1])[:truncate_idx[0]], torch.eye(A.shape[1])[truncate_idx[1]+1:]), dim=0).to(self._device)

    @staticmethod
    def b(in_val):
        return in_val

    @property
    def c(self):
        return self._c

    @property
    def A(self):
        return self._A

    @property
    def const_num(self):
        return self._const_num

    @property
    def var_num(self):
        return self._var_num

    @property
    def free_idx(self):
        return self._free_idx

    @property
    def free_num(self):
        return self._free_num

    @property
    def truncate_idx(self):
        return self._truncate_idx

    @property
    def free_W(self):
        return self._free_W

    @property
    def s_W(self):
        return self._s_W

    @property
    def mutable_W(self):
        return self._mutable_W

    @property
    def immutable_W(self):
        return self._immutable_W

    @property
    def device(self):
        return self._device

    # def get_free(self, x):
    #     return x @ self._free_W.t()
    #
    # def get_s(self, x):
    #     return x @ self._s_W.t()

    def get_mutable(self, x):
        return x @ self._mutable_W.t()

    def get_immutable(self, x):
        return x @ self._immutable_W.t()

    def obj_fn(self, x, in_val=None):
        return x @ self.c

    def optimality_gap(self, x, target, in_val=None):
        predicted_obj = x @ self.c
        optimality_gap = (predicted_obj - target) / target
        return optimality_gap

    def eq_residual(self, x, b):
        return x @ self.A.t() - b

    def ineq_residual(self, x):
        # fx, s = torch.split(x, [self._free_num, self._var_num - self._free_num], dim=-1)
        # return torch.relu(-s)
        return torch.relu(-x[:, self.free_num:])


class DualLP:
    """
        maximize_y c^T y
        s.t.       Ay = b
                   s >= 0
    where -c (b_primal) will be inputs and y will be outputs in the neural network
    """

    def __init__(self, b, A, free_idx, truncate_idx):
        self.name = 'dual_lp'
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._b = b
        self._A = A

        self._const_num = A.shape[0]
        self._var_num = A.shape[1]
        self._free_idx = free_idx  # free_idx[0] = 0, free_idx[1] = free_num - 1, made by default
        self._free_num = free_idx[1] + 1
        self._truncate_idx = truncate_idx

        self._free_W = torch.eye(A.shape[1])[free_idx[0]:free_idx[1]].to(self._device)
        self._s_W = torch.cat((torch.eye(A.shape[1])[:free_idx[0]], torch.eye(A.shape[1])[free_idx[1]+1:]), dim=0).to(self._device)

        self._mutable_W = torch.eye(A.shape[1])[truncate_idx[0]:truncate_idx[1]].to(self._device)
        self._immutable_W = torch.cat((torch.eye(A.shape[1])[:truncate_idx[0]], torch.eye(A.shape[1])[truncate_idx[1]+1:]), dim=0).to(self._device)

    def b(self, in_val=None):
        return self._b

    @property
    def A(self):
        return self._A

    @property
    def const_num(self):
        return self._const_num

    @property
    def var_num(self):
        return self._var_num

    @property
    def free_idx(self):
        return self._free_idx

    @property
    def free_num(self):
        return self._free_num

    @property
    def truncate_idx(self):
        return self._truncate_idx

    @property
    def free_W(self):
        return self._free_W

    @property
    def s_W(self):
        return self._s_W

    @property
    def mutable_W(self):
        return self._mutable_W

    @property
    def immutable_W(self):
        return self._immutable_W

    @property
    def device(self):
        return self._device

    # def get_free(self, x):
    #     return x @ self._free_W.t()
    #
    # def get_s(self, x):
    #     return x @ self._s_W.t()

    def get_mutable(self, x):
        return x @ self._mutable_W.t()

    def get_immutable(self, x):
        return x @ self._immutable_W.t()

    @staticmethod
    def obj_fn(y, c):
        # y, b_primal are passed in, b_primal = -c in maximize_y c^T y and c in minimize_x - c^T x
        return torch.sum(y * c, dim=-1)

    @staticmethod
    def optimality_gap(y, target, c=None):
        predicted_obj = torch.sum(y * c, dim=-1)
        optimality_gap = (predicted_obj + target) / target
        return optimality_gap

    def eq_residual(self, y, in_val=None):
        return y @ self.A.t() - self.b(None)

    def ineq_residual(self, y):
        # fx, s = torch.split(y, [self._free_num, self._var_num - self._free_num], dim=-1)
        # return torch.relu(-s)
        return torch.relu(-y[:, self.free_num:])




