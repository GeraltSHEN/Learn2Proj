import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

import numpy as np
from scipy.linalg import qr

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
        """
        DC3 set up: A_old x_old = b_old and G_old x_old <= h_old
        Our Data structure: Ax = b and s >= 0
        Transformation:
            A_old   0
        A =
            G_old   I

            x_old
        x =
            s

            b_old
        b =
            h_old

        In DC3 experiments,  A_old x_old = b_old and G_old x_old <= h_old
        when changing feature is 'b',
        we use A as A_old, x as x_old, b as b_old, Mat(nonnegative_mask) as G_old, 0 as h_old
        A is stored, partial x: (constr_num, ) is predicted, other x: (var_num - constr_num, ) is completed
        b is changing and given, Mat(nonnegative_mask) is stored, 0 is stored

        when changing feature is 'A',
        we use A_old as A_old, x_old as x_old, b_old as b_old, G_old as G_old, h_old as h_old
        A_old is stored, partial x_old: (A_old.shape[0], ) is predicted, other x_old: (A_old.shape[1] - A_old.shape[0], ) is completed
        b_old is stored, G_old is changing and given, h_old is stored
        """

        super(DC3, self).__init__()
        self.name = 'DC3'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.momentum = momentum
        self.old_x_step = 0
        self.changing_feature = changing_feature

        A_cpu = (A.to_dense() if A.is_sparse else A).cpu()
        self.constr_num, self.var_num = A_cpu.shape

        if changing_feature == 'b':
            cols = nonnegative_mask.nonzero(as_tuple=True)[0]
            rows = torch.arange(len(cols), device=self.device)
            G = torch.zeros((len(rows), self.var_num), device=self.device)
            G[rows, cols] = 1.0
            self.register_buffer('G', G)
            self.register_buffer('h', torch.tensor(0., device=self.device))
        elif changing_feature == 'A':
            raise NotImplementedError
        else:
            raise ValueError("Invalid changing feature. Must be 'b' or 'A'.")

        with torch.no_grad():  # no autograd needed
            Q, R, P = qr(A_cpu.numpy(), pivoting=True)
        P = torch.from_numpy(P).to(self.device)

        r = np.linalg.matrix_rank(A_cpu.numpy())
        self.register_buffer('_other_vars', P[:r])
        self.register_buffer('_partial_vars', P[r:])

        A_other = A_cpu[:, P[:r]].to(self.device)
        A_partial = A_cpu[:, P[r:]].to(self.device)

        self.register_buffer('_A_other_inv', torch.inverse(A_other))
        self.register_buffer('_A_partial', A_partial)


    def reset_old_x_step(self):
        self.old_x_step = 0

    def update_bh(self, b):
        if self.changing_feature == 'A':
            self.b = b[:, self.constr_num]
            self.h = b[:, self.constr_num:]

    def update_G(self, A):
        if self.changing_feature == 'A':
            A = A.to_dense()

    def complete(self, x, b):
        bsz = x.shape[0]
        complete_x = torch.zeros(bsz, self.var_num, device=self.device)
        complete_x[:, self._partial_vars] = x
        if self.changing_feature == 'A':
            b = self.b
        complete_x[:, self._other_vars] = (b - x @ self._A_partial.T) @ self._A_other_inv.T
        return complete_x

    def ineq_partial_grad(self, x, b, G):
        bsz = x.shape[0]
        if self.changing_feature == 'b':
            G = self.G.repeat(bsz, 1)
        G_effective = G[:, self._partial_vars] - G[:, self._other_vars] @ (self._A_other_inv @ self._A_partial)
        h_effective = self.h - (b @ self._A_other_inv.T) @ G[:, self._other_vars].T
        grad = 2 * torch.clamp(x[:, self._partial_vars] @ G_effective.T - h_effective, 0) @ G_effective
        x = torch.zeros(bsz, self.var_num, device=self.device)
        x[:, self._partial_vars] = grad
        x[:, self._other_vars] = - (grad @ self._A_partial.T) @ self._A_other_inv.T
        return x

    def forward(self, x, b, G_old):
        x_step = self.ineq_partial_grad(x, b, G_old)
        new_x_step = self.lr * x_step + self.momentum * self.old_x_step
        x = x - new_x_step
        self.old_x_step = new_x_step
        return x

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
    def __init__(self, nonnegative_mask, eq_weight, eq_bias_transform, ldr_weight, ldr_bias, ldr_temp):
        super(LDRPM, self).__init__()
        self.name = 'LDRPM'
        self.nonnegative_mask = nonnegative_mask.bool()
        self.eq_projector = Projector(weight=eq_weight, bias_transform=eq_bias_transform)
        self.ldr_projector = Projector(weight=ldr_weight, bias=ldr_bias)
        self.x_LDR = ldr_bias
        self.ldr_temp = ldr_temp

    def update_ldr_ref(self, features):
        with torch.no_grad():
            self.x_LDR = self.ldr_projector(features)
            # check if there is 0 in the x_LDR where self.nonnegative_mask is True
            if torch.any((self.x_LDR == 0) & self.nonnegative_mask.bool()):
                print('Warning: x_LDR has 0 in the nonnegative region')

    def forward(self, x):
        x_eq = self.eq_projector(x)
        s = (self.x_LDR - x_eq).clamp_min(torch.finfo(x_eq.dtype).eps)
        alphas = -x_eq / s
        mask = (x_eq < 0) & self.nonnegative_mask.bool()

        if self.training:
            logits = alphas * self.ldr_temp
            neg_inf = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~mask, neg_inf)
            weights = torch.softmax(logits, dim=-1)
            weights = weights * mask  # zero out weights where mask is False (feasible region)
            denom = weights.sum(dim=-1, keepdim=True)
            # If denom is zero (-> all weights zero -> mask has all False -> all entries feasible), weights remain unchanged to avoid division by zero
            weights = torch.where(denom > 0, weights / denom, weights)
            alpha = (alphas * weights).sum(-1)
        else:
            masked_alphas = alphas * mask
            alpha = torch.max(masked_alphas, dim=-1).values

        # masked_alphas = alphas * mask
        # alpha = torch.max(masked_alphas, dim=-1).values

        self.alpha = alpha

        x_star = self.x_LDR * alpha.unsqueeze(-1) + x_eq * (1.0 - alpha).unsqueeze(-1)
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
            # if self.changing_feature == 'A':
            #     self.algo.eq_projector.update_weight_and_bias_transform(A)
            self.algo.eq_projector.update_bias(b)
        elif self.algo_name == 'LDRPM':
            # if self.changing_feature == 'A':
            #     self.algo.eq_projector.update_weight_and_bias_transform(A)
            self.algo.eq_projector.update_bias(b)
            self.algo.update_ldr_ref(feature)

        self.iters = 0

        self.eq_epsilon, self.ineq_epsilon = self.stopping_criterion(x, A, b, nonnegative_mask)
        while ((self.eq_epsilon.mean() > self.eq_tol or self.ineq_epsilon.mean() > self.ineq_tol)
               and self.iters < self.max_iters):

            if self.algo_name == 'DC3':
                x = self.algo(x, b, None)  # placeholder for G_old
            elif self.algo_name == 'POCS':
                x = self.algo(x)
            elif self.algo_name == 'LDRPM':
                x = self.algo(x)
                self.iters += 1
                self.eq_epsilon, self.ineq_epsilon = self.stopping_criterion(x, A, b, nonnegative_mask)
                break

            self.iters += 1
            self.eq_epsilon, self.ineq_epsilon = self.stopping_criterion(x, A, b, nonnegative_mask)
            #print(f'iter: {self.iters}, eq_epsilon: {self.eq_epsilon.mean()}, ineq_epsilon: {self.ineq_epsilon.mean()}')
        return x

    @staticmethod
    def stopping_criterion(x, A, b, nonnegative_mask):
        with torch.no_grad():
            eq_residual = (A @ x.flatten() - b.flatten()).view(-1, b.shape[-1])
            ineq_residual = torch.relu(-x[:, nonnegative_mask])

            eq_violation = torch.norm(eq_residual, p=2, dim=-1)
            ineq_violation = torch.norm(ineq_residual, p=2, dim=-1)

            eq_epsilon = eq_violation / (1 + torch.norm(b, p=2, dim=-1))  # scaled by the norm of b
            ineq_epsilon = ineq_violation  # no scaling because rhs is 0
            return eq_epsilon, ineq_epsilon









# class OPTNET(nn.Module):
#     def __init__(self, nonnegative_mask, constr_num, var_num):
#         super().__init__()
#         self.name = 'OPTNET'
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.nonnegative_mask = nonnegative_mask
#         self.constr_num = constr_num
#         self.var_num = var_num
#
#         cols = nonnegative_mask.nonzero(as_tuple=True)[0]
#         rows = torch.arange(len(cols))
#         _G = torch.zeros((len(rows), var_num))
#         _G[rows, cols] = 1.0
#         self.G = _G.to(self.device)
#         self.h = 0  # torch.zeros(int(nonnegative_mask.sum().item()))
#
#         x = cp.Variable(var_num)
#         x0 = cp.Parameter(var_num)
#         A = cp.Parameter((constr_num, var_num))
#         b = cp.Parameter(constr_num)
#
#         constraints = [A @ x == b, _G.numpy() @ x >= 0]
#         objective = cp.Minimize(cp.sum_squares(x - x0))
#         problem = cp.Problem(objective, constraints)
#         self.proj_layer = CvxpyLayer(problem, variables=[x],
#                                      parameters=[x0, b, A])
#         # test the accuracy of the projection layer
#         # constraints = [x >= self.h]
#         # objective = cp.Minimize(cp.sum_squares(x - x0))
#         # problem = cp.Problem(objective, constraints)
#         # self.proj_layer = CvxpyLayer(problem, variables=[x],
#         #                              parameters=[x0])
#
#     def forward(self, x, b, A):
#         bsz = x.shape[0]
#
#         # start_time = time.time()
#         # cp_x = cp.Variable(self.var_num * bsz)
#         # cp_x0 = cp.Parameter(self.var_num * bsz)
#         # cp_A = cp.Parameter((self.constr_num * bsz, self.var_num * bsz))
#         # cp_b = cp.Parameter(self.constr_num * bsz)
#         # cp_G = torch.block_diag(*[self.G for _ in range(bsz)]).to(self.device)
#         # cp_h = torch.zeros(int(self.nonnegative_mask.sum().item()) * bsz, device=self.device)
#         #
#         # constraints = [cp_A @ cp_x == cp_b, cp_G @ cp_x >= 0]
#         # objective = cp.Minimize(cp.sum_squares(cp_x - cp_x0))
#         # problem = cp.Problem(objective, constraints)
#         # proj_layer = CvxpyLayer(problem, variables=[cp_x], parameters=[cp_x0, cp_b, cp_A])
#         # print(f"Time to set up CVXPY problem: {time.time() - start_time:.4f} seconds")
#         #
#         # start_time = time.time()
#         # x_proj = proj_layer(x.flatten(), b.flatten(), A.to_dense())[0]
#         # print(f"Time to solve CVXPY problem: {time.time() - start_time:.4f} seconds")
#         # return x_proj.view(bsz, self.var_num)
#
#         # not parallelized
#         xs = []
#         A = A.to_dense()
#         for i in range(bsz):
#             x0 = x[i]
#             b_i = b[i]
#             A_i = A[i*self.constr_num:(i+1)*self.constr_num, i*self.var_num:(i+1)*self.var_num]
#             x_proj = self.proj_layer(x0, b_i, A_i)[0]
#             xs.append(x_proj)
#         return torch.stack(xs)












