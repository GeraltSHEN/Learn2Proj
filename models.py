import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class MLP(nn.Module):
    def __init__(self,
                 features_dim,
                 hidden_dims,
                 out_dim,
                 act_cls=nn.ReLU,
                 batch_norm=False):
        super().__init__()
        layers = []
        layers_dims = [features_dim] + hidden_dims + [out_dim]

        for k in range(len(layers_dims) - 1):
            layers.append(nn.Linear(in_features=layers_dims[k], out_features=layers_dims[k + 1]))
            if k < len(layers_dims) - 2:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(layers_dims[k + 1]))
                layers.append(act_cls())

        self.latent_dim = layers_dims[-1]
        self.mlp = nn.Sequential(*layers)

    def forward(self, features):
        x = self.mlp(features)
        return x


class DC3(nn.Module):
    def __init__(self, A, nonnegative_mask, lr, momentum, changing_feature):
        super(DC3, self).__init__()
        self.name = 'DC3'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.momentum = momentum
        self.old_x_step = 0

        self.changing_feature = changing_feature

        _A = A.to_dense()
        self.constr_num = _A.shape[0]
        self.var_num = _A.shape[1]

        self.nonnegative_mask = nonnegative_mask
        cols = nonnegative_mask.nonzero(as_tuple=True)[0]
        rows = torch.arange(len(cols))
        _G = torch.zeros((len(rows), self.var_num), device=self.device)
        _G[rows, cols] = 1.0
        self.G = _G
        self.h = 0

        det = 0
        i = 0
        while abs(det) < 1e-8 and i < 100:
            self._partial_vars = np.random.choice(self.var_num, self.var_num - self.constr_num, replace=False)
            self._other_vars = np.setdiff1d(np.arange(self.var_num), self._partial_vars)
            det = torch.det(_A[:, self._other_vars])
            i += 1
            print(f'det: {det}')
        if i == 100:
            raise Exception
        else:
            self._A_partial = _A[:, self._partial_vars].requires_grad_(False)
            self._A_other_inv = torch.inverse(_A[:, self._other_vars]).requires_grad_(False)

    def reset_old_x_step(self):
        self.old_x_step = 0

    def complete(self, x, b):
        complete_x = torch.zeros(b.shape[0], self.var_num, device=self.device)
        complete_x[:, self.partial_vars] = x
        complete_x[:, self.other_vars] = (b - x @ self._A_partial.T) @ self._A_other_inv.T
        return complete_x

    def ineq_partial_grad(self, x, b):
        G_effective = self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (self._A_other_inv @ self._A_partial)
        h_effective = self.h - (b @ self._A_other_inv.T) @ self.G[:, self.other_vars].T
        grad = 2 * torch.clamp(x[:, self.partial_vars] @ G_effective.T - h_effective, 0) @ G_effective
        x = torch.zeros(b.shape[0], self.var_num, device=self.device)
        x[:, self.partial_vars] = grad
        x[:, self.other_vars] = - (grad @ self._A_partial.T) @ self._A_other_inv.T
        return x

    def ineq_grad(self, x):
        return 2 * torch.clamp(x@self.G.T - self.h, 0) @ self.G

    def forward(self, x, b):
        if self.changing_feature == 'A':
            x_step = self.ineq_grad(x)
        elif self.changing_feature == 'b':
            x_step = self.ineq_partial_grad(x, b)
        else:
            raise ValueError("Invalid changing_feature. Must be either 'A' or 'b'.")
        new_x_step = self.lr * x_step + self.momentum * self.old_x_step
        x = x - new_x_step
        self.old_x_step = new_x_step
        return x


class OPTNET(nn.Module):
    def __init__(self, nonnegative_mask, constr_num, var_num):
        super().__init__()
        self.name = 'OPTNET'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nonnegative_mask = nonnegative_mask
        self.constr_num = constr_num
        self.var_num = var_num

        cols = nonnegative_mask.nonzero(as_tuple=True)[0]
        rows = torch.arange(len(cols))
        _G = torch.zeros((len(rows), var_num), device=self.device)
        _G[rows, cols] = 1.0
        self.G = _G
        self.h = 0  # torch.zeros(int(nonnegative_mask.sum().item()))

        x = cp.Variable(var_num)
        x0 = cp.Parameter(var_num)
        A = cp.Parameter((constr_num, var_num))
        b = cp.Parameter(constr_num)

        constraints = [A @ x == b, self.G @ x >= self.h]
        objective = cp.Minimize(cp.sum_squares(x - x0))
        problem = cp.Problem(objective, constraints)
        self.proj_layer = CvxpyLayer(problem, variables=[x],
                                     parameters=[x0, b, A])
        # test the accuracy of the projection layer
        # constraints = [x >= self.h]
        # objective = cp.Minimize(cp.sum_squares(x - x0))
        # problem = cp.Problem(objective, constraints)
        # self.proj_layer = CvxpyLayer(problem, variables=[x],
        #                              parameters=[x0])

    def forward(self, x, b, A):
        bsz = x.shape[0]

        # start_time = time.time()
        # cp_x = cp.Variable(self.var_num * bsz)
        # cp_x0 = cp.Parameter(self.var_num * bsz)
        # cp_A = cp.Parameter((self.constr_num * bsz, self.var_num * bsz))
        # cp_b = cp.Parameter(self.constr_num * bsz)
        # cp_G = torch.block_diag(*[self.G for _ in range(bsz)]).to(self.device)
        # cp_h = torch.zeros(int(self.nonnegative_mask.sum().item()) * bsz, device=self.device)
        #
        # constraints = [cp_A @ cp_x == cp_b, cp_G @ cp_x >= 0]
        # objective = cp.Minimize(cp.sum_squares(cp_x - cp_x0))
        # problem = cp.Problem(objective, constraints)
        # proj_layer = CvxpyLayer(problem, variables=[cp_x], parameters=[cp_x0, cp_b, cp_A])
        # print(f"Time to set up CVXPY problem: {time.time() - start_time:.4f} seconds")
        #
        # start_time = time.time()
        # x_proj = proj_layer(x.flatten(), b.flatten(), A.to_dense())[0]
        # print(f"Time to solve CVXPY problem: {time.time() - start_time:.4f} seconds")
        # return x_proj.view(bsz, self.var_num)

        # not parallelized
        xs = []
        A = A.to_dense()
        for i in range(bsz):
            x0 = x[i]
            b_i = b[i]
            A_i = A[i*self.constr_num:(i+1)*self.constr_num, i*self.var_num:(i+1)*self.var_num]
            x_proj = self.proj_layer(x0, b_i, A_i)[0]
            xs.append(x_proj)
        return torch.stack(xs)


class Projector(nn.Module):
    def __init__(self, weight, bias=None, bias_transform=None):
        super(Projector, self).__init__()

        if bias is not None:
            assert bias_transform is None
            self.bias = bias.requires_grad_(False)
        if bias_transform is not None:
            assert bias is None
            self.bias_transform = bias_transform.t().requires_grad_(False)

        self.weight = weight.t().requires_grad_(False)

    def forward(self, x):
        return self.bias + x @ self.weight

    def update_bias(self, b):
        with torch.no_grad():
            self.bias = b @ self.bias_transform

    def update_weight_and_bias_transform(self, A):
        A = A.to_sparse if not A.is_sparse else A
        with torch.no_grad():
            PD = torch.sparse.mm(A, A.t())
            chunk = torch.sparse.mm(A.t(), torch.inverse(PD.to_dense()))
            self.weight = (torch.eye(A.shape[-1]) - torch.sparse.mm(chunk, A)).t().requires_grad_(False)
            self.bias_transform = chunk.t().requires_grad_(False)


class POCS(nn.Module):
    def __init__(self, nonnegative_mask, eq_weight, eq_bias_transform):
        super(POCS, self).__init__()
        self.name = 'POCS'
        self.nonnegative_mask = nonnegative_mask
        self.eq_projector = Projector(weight=eq_weight, bias_transform=eq_bias_transform)

    def forward(self, x):
        x = torch.where(self.nonnegative_mask, torch.clamp(x, min=0), x)
        x_eq = self.eq_projector(x)
        return x_eq


class LDRPM(nn.Module):
    def __init__(self, nonnegative_mask, eq_weight, eq_bias_transform, ldr_weight, ldr_bias):
        super(LDRPM, self).__init__()
        self.name = 'LDRPM'
        self.nonnegative_mask = nonnegative_mask
        self.eq_projector = Projector(weight=eq_weight, bias_transform=eq_bias_transform)
        self.ldr_projector = Projector(weight=ldr_weight, bias=ldr_bias)
        self.x_LDR = ldr_bias

    def update_ldr_ref(self, features):
        with torch.no_grad():
            self.x_LDR = self.ldr_projector(features)

    def forward(self, x):
        x_eq = self.eq_projector(x)
        s = self.x_LDR - x_eq
        alphas = - x_eq / (s + 1e-8)
        mask = (x_eq < 0) * self.nonnegative_mask
        alpha = torch.max(alphas * mask, dim=-1).values
        x_star = self.x_LDR * alpha.unsqueeze(-1) + x_eq * (1 - alpha).unsqueeze(-1)
        return x_star


class FeasibilityNet(nn.Module):
    def __init__(self, algo, eq_tol, ineq_tol, max_iters, changing_feature):
        super(FeasibilityNet, self).__init__()
        self.algo_name = algo.name
        self.algo = algo
        self.eq_tol = eq_tol
        self.ineq_tol = ineq_tol
        self.max_iters = max_iters
        self.changing_feature = changing_feature

        self.eq_epsilon = None
        self.ineq_epsilon = None
        self.iters = 0

    def forward(self, x, A, b, nonnegative_mask, feature):
        if self.algo_name == 'DC3':
            if self.changing_feature == 'b':
                x = self.algo.complete(x, b)  # complete
            self.algo.reset_old_x_step()
        elif self.algo_name == 'POCS':
            if self.changing_feature == 'A':
                self.algo.eq_projector.update_weight_and_bias_transform(A)
            self.algo.eq_projector.update_bias(b)
        elif self.algo_name == 'LDRPM':
            if self.changing_feature == 'A':
                self.algo.eq_projector.update_weight_and_bias_transform(A)
            self.algo.eq_projector.update_bias(b)
            self.algo.update_ldr_ref(feature)

        self.iters = 0

        # x_eq = self.algo.eq_projector(x)
        # s = self.algo.x_LDR - x_eq
        # alphas = - x_eq / (s + 1e-8)
        # mask = (x_eq < 0) * self.algo.nonnegative_mask
        # alpha = torch.max(alphas * mask, dim=-1).values
        # x_star = self.algo.x_LDR * alpha.unsqueeze(-1) + x_eq * (1 - alpha).unsqueeze(-1)
        #
        # def ff(x):
        #     eq_residual = (A @ x.flatten() - b.flatten()).view(-1, b.shape[-1])
        #     ineq_residual = torch.relu(-x[:, nonnegative_mask])
        #     eq_violation = torch.norm(eq_residual, p=2, dim=-1)
        #     ineq_violation = torch.norm(ineq_residual, p=2, dim=-1)
        #     return eq_violation, ineq_violation
        #
        # print("debug")

        self.eq_epsilon, self.ineq_epsilon = self.stopping_criterion(x, A, b, nonnegative_mask)
        while ((self.eq_epsilon.mean() > self.eq_tol or self.ineq_epsilon.mean() > self.ineq_tol)
               and self.iters < self.max_iters):

            if self.algo_name == 'DC3':
                x = self.algo(x, b)
            elif self.algo_name == 'OPTNET':
                x = self.algo(x, b, A)
                self.iters += 1
                self.eq_epsilon, self.ineq_epsilon = self.stopping_criterion(x, A, b, nonnegative_mask)
                break
            elif self.algo_name == 'POCS':
                x = self.algo(x)
            elif self.algo_name == 'LDRPM':
                x = self.algo(x)
                self.iters += 1
                self.eq_epsilon, self.ineq_epsilon = self.stopping_criterion(x, A, b, nonnegative_mask)
                break

            self.iters += 1
            self.eq_epsilon, self.ineq_epsilon = self.stopping_criterion(x, A, b, nonnegative_mask)

        return x

    @staticmethod
    def stopping_criterion(x, A, b, nonnegative_mask):
        with torch.no_grad():
            # eq_residual = x @ A.t() - b
            eq_residual = (A @ x.flatten() - b.flatten()).view(-1, b.shape[-1])
            ineq_residual = torch.relu(-x[:, nonnegative_mask])

            eq_violation = torch.norm(eq_residual, p=2, dim=-1)
            ineq_violation = torch.norm(ineq_residual, p=2, dim=-1)

            eq_epsilon = eq_violation / (1 + torch.norm(b, p=2, dim=-1))  # scaled by the norm of b
            ineq_epsilon = ineq_violation  # no scaling because rhs is 0
            return eq_epsilon, ineq_epsilon










