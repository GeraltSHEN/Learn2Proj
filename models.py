import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

        A = A.to_dense() if A.is_sparse else A
        A_other = A[:, P[:r]]
        A_partial = A[:, P[r:]]

        self.register_buffer('_A_other_inv', torch.inverse(A_other))
        self.register_buffer('_A_partial', A_partial)

        G_effective = G[:, self._partial_vars] - G[:, self._other_vars] @ (self._A_other_inv @ self._A_partial)
        self.register_buffer('_G_effective', G_effective)
        G_other_t = G[:, self._other_vars].T
        self.register_buffer('_G_other_t', G_other_t)


    def reset_old_x_step(self):
        self.old_x_step = 0

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
            G = self.G
            G_effective = self._G_effective
            G_other_t = self._G_other_t
        h_effective = self.h - (b @ self._A_other_inv.T) @ G_other_t
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

    def forward(self, x):
        x_eq = self.eq_projector(x)
        s = (self.x_LDR - x_eq).clamp_min(torch.finfo(x_eq.dtype).eps)
        alphas = -x_eq / s
        mask = (x_eq < 0) & self.nonnegative_mask.bool()
        masked_alphas = alphas * mask
        alpha = torch.max(masked_alphas, dim=-1).values

        self.alpha = alpha

        x_star = self.x_LDR * alpha.unsqueeze(-1) + x_eq * (1.0 - alpha).unsqueeze(-1)
        return x_star


class DC3LHS(nn.Module):
    def __init__(self, A, h, b, lr, momentum):
        super(DC3LHS, self).__init__()
        self.name = 'DC3LHS'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.momentum = momentum
        self.old_x_step = 0

        A_cpu = (A.to_dense() if A.is_sparse else A).cpu()
        self.constr_num, self.var_num = A_cpu.shape
        self.register_buffer('h', h)
        self.register_buffer('b', b)

        with torch.no_grad():
            Q, R, P = qr(A_cpu.numpy(), pivoting=True)
        P = torch.from_numpy(P).to(self.device)

        r = np.linalg.matrix_rank(A_cpu.numpy())
        self.register_buffer('_other_vars', P[:r])
        self.register_buffer('_partial_vars', P[r:])

        A = A.to_dense() if A.is_sparse else A
        A_other = A[:, P[:r]]
        A_partial = A[:, P[r:]]

        self.register_buffer('_A_other_inv', torch.inverse(A_other))
        self.register_buffer('_A_partial', A_partial)


    def reset_old_x_step(self):
        self.old_x_step = 0

    def complete(self, x):
        bsz = x.shape[0]
        complete_x = torch.zeros(bsz, self.var_num, device=self.device)
        complete_x[:, self._partial_vars] = x
        complete_x[:, self._other_vars] = (self.b - x @ self._A_partial.T) @ self._A_other_inv.T
        return complete_x

    def ineq_partial_grad(self, x, G):
        bsz = x.shape[0]
        ineq_num = self.h.numel()
        var_num = self.var_num

        G_4d = G_blk.view(bsz, ineq_num, bsz, var_num)
        idx = torch.arange(bsz, device=device)
        G = G_4d[idx, :, idx, :]
        G_p, G_o = G[..., self._partial_vars], G[..., self._other_vars]

        # ------------------------------------------------------------------
        # 2. Build the “effective” G that eliminates x_other
        # ------------------------------------------------------------------
        C = self._A_other_inv @ self._A_partial  # (|other|, |partial|)
        G_eff = G_p - torch.matmul(G_o, C)  # (bsz, ineq, |partial|)

        # ------------------------------------------------------------------
        # 3. Build the effective right-hand side h̃ = h − (b Aₒ⁻¹) Gᵒᵀ
        # ------------------------------------------------------------------
        b_eff = torch.matmul(self.b, self._A_other_inv.T)  # (bsz, |other|)
        term = torch.bmm(b_eff.unsqueeze(1),  # (bsz,1,|other|)
                         G_o.transpose(1, 2))  # (bsz,|other|,ineq)
        h_eff = self.h.to(device).unsqueeze(0) - term.squeeze(1)  # (bsz, ineq)

        # ------------------------------------------------------------------
        # 4. Positive-part residuals r⁺ = max(0, x_p G_effᵀ − h̃)
        # ------------------------------------------------------------------
        x_p = x[:, p_idx]  # (bsz, |partial|)
        res = torch.bmm(x_p.unsqueeze(1),  # (bsz,1,|partial|)
                        G_eff.transpose(1, 2)).squeeze(1) - h_eff
        r_pos = torch.clamp(res, min=0.)  # (bsz, ineq)

        # ------------------------------------------------------------------
        # 5. Gradient wrt x_partial: 2 r⁺ G_eff
        # ------------------------------------------------------------------
        g_p = 2.0 * torch.bmm(r_pos.unsqueeze(1), G_eff).squeeze(1)  # (bsz, |partial|)

        # ------------------------------------------------------------------
        # 6. Back-propagate to “other” variables via the elimination rule
        # ------------------------------------------------------------------
        g_o = -torch.matmul(torch.matmul(g_p, self._A_partial.T), self._A_other_inv.T)  # (bsz, |other|)

        # ------------------------------------------------------------------
        # 7. Assemble full gradient
        # ------------------------------------------------------------------
        grad_full = torch.zeros(bsz, var_num, device=device)
        grad_full[:, p_idx] = g_p
        grad_full[:, o_idx] = g_o
        return grad_full

    def forward(self, x, G):
        x_step = self.ineq_partial_grad(x, G)
        new_x_step = self.lr * x_step + self.momentum * self.old_x_step
        x = x - new_x_step
        self.old_x_step = new_x_step
        return x


class LDRPMLHS(nn.Module):
    def __init__(self, h, eq_weight, eq_bias, ldr_weight, ldr_bias, S_scale, nonnegative_mask):
        # todo: remember to pass in h and G as -h and -G
        super(LDRPMLHS, self).__init__()
        self.name = 'LDRPMLHS'
        self.register_buffer('h', h)
        self.eq_projector = Projector(weight=eq_weight, bias=eq_bias)
        self.ldr_projector = Projector(weight=ldr_weight, bias=ldr_bias)

        self.ldr_bias = ldr_bias.requires_grad_(False)
        self.ldr_weight = ldr_weight.t().requires_grad_(False)
        self.S_scale = S_scale.requires_grad_(False)
        self.nonnegative_mask = nonnegative_mask

        self.x_LDR = ldr_bias


    def update_ldr_ref(self, features):
        with torch.no_grad():
            eq_part = self.ldr_bias + features @ self.ldr_weight
            ones_features = torch.cat((torch.ones(features.shape[0], 1), features), dim=1)
            ineq_part = torch.einsum('bd,ndm->bnm', ones_features, self.S_scale)
            ineq_part = torch.einsum('bnm,bm->bn', ineq_part, ones_features)
            self.x_LDR = torch.cat((eq_part, ineq_part), dim=1)

    def forward(self, x, G):
        bsz = x.shape[0]

        y = x[:, ~self.nonnegative_mask.bool()]
        y_eq = self.eq_projector(y)
        s_eq = self.h - (G @ y_eq.unsqueeze(-1)).squeeze(-1)

        y_LDR = self.x_LDR[:, ~self.nonnegative_mask.bool()]
        s_LDR = self.x_LDR[:, self.nonnegative_mask.bool()]
        s = (s_LDR - s_eq).clamp_min(torch.finfo(x.dtype).eps)
        alphas = -s_eq / s

        mask = s_eq < 0
        masked_alphas = alphas * mask
        alpha = torch.max(masked_alphas, dim=-1).values

        self.alpha = alpha

        y_star = self.x_LDR[:, ~self.nonnegative_mask.bool()] * alpha.unsqueeze(-1) + y_eq * (1.0 - alpha).unsqueeze(-1)
        s_star = s_LDR * alpha.unsqueeze(-1) + s_eq * (1.0 - alpha).unsqueeze(-1)
        x_star = torch.cat([y_star, s_star], dim=1)
        return x_star


        # x_eq_partial = x_eq[:, ~self.nonnegative_mask.bool()]
        # x_LDR_partial = x_LDR[:, ~self.nonnegative_mask.bool()]
        #
        # s = (G @ (x_LDR_partial - x_eq_partial).unsqueeze(-1)).squeeze(-1).clamp_min(torch.finfo(x_eq.dtype).eps)
        # alphas = (self.h - (G @ x_eq_partial.unsqueeze(-1)).squeeze(-1)) / s
        #
        # mask = (G @ x_eq_partial.unsqueeze(-1)).squeeze(-1) < self.h
        #
        # masked_alphas = alphas * mask
        # alpha = torch.max(masked_alphas, dim=-1).values
        #
        # self.alpha = alpha
        #
        # x_star_partial = x_LDR_partial * alpha.unsqueeze(-1) + x_eq_partial * (1.0 - alpha).unsqueeze(-1)
        # x_eq[:, ~self.nonnegative_mask.bool()] = x_star_partial
        # return x_eq











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

    def forward(self, x, A, b, nonnegative_mask, feature, G):
        if self.algo_name == 'DC3':
            x = self.algo.complete(x, b)  # complete
            self.algo.reset_old_x_step()
        elif self.algo_name == 'DC3LHS':
            x = self.algo.complete(x)
            self.algo.reset_old_x_step()
        elif self.algo_name == 'POCS':
            self.algo.eq_projector.update_bias(b)
        elif self.algo_name == 'LDRPM':
            self.algo.eq_projector.update_bias(b)
            self.algo.update_ldr_ref(feature)
        elif self.algo_name == 'LDRPMLHS':
            self.algo.update_ldr_ref(feature)

        self.iters = 0

        eq_residual = (A @ self.algo.x_LDR.flatten() - b.flatten()).view(-1, b.shape[-1])
        ineq_residual = torch.relu(-self.algo.x_LDR[:, nonnegative_mask])

        self.eq_epsilon, self.ineq_epsilon = self.stopping_criterion(x, A, b, nonnegative_mask)
        while ((self.eq_epsilon.mean() > self.eq_tol or self.ineq_epsilon.mean() > self.ineq_tol)
               and self.iters < self.max_iters):

            if self.algo_name == 'DC3':
                x = self.algo(x, b, None)
            elif self.algo_name == 'DC3LHS':
                x = self.algo(x, G)
            elif self.algo_name == 'POCS':
                x = self.algo(x)
            elif self.algo_name == 'LDRPM':
                x = self.algo(x)
                self.iters += 1
                self.eq_epsilon, self.ineq_epsilon = self.stopping_criterion(x, A, b, nonnegative_mask)
                break
            elif self.algo_name == 'LDRPMLHS':
                x = self.algo(x, G)
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