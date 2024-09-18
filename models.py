import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

#import torch.autograd.Function as Function

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#torch.set_default_dtype(torch.float64)


class OptimalityLayers(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, output_dim,
                 truncate_idx):
        super(OptimalityLayers, self).__init__()

        self.truncate_idx = truncate_idx

        # optimality layers
        self.optimality_layers = nn.ModuleList()
        self.optimality_layers.append(nn.Linear(input_dim, hidden_dim))
        self.optimality_layers.append(nn.LayerNorm(hidden_dim))  # add layer norm
        for _ in range(hidden_num - 1):
            self.optimality_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.optimality_layers.append(nn.LayerNorm(hidden_dim))  # add layer norm
        self.optimality_layers.append(nn.Linear(hidden_dim, output_dim))

    def initialize_with_zeros(self):
        # initialize all layers with zeros
        for layer in self.optimality_layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def truncate(self, b_primal):
        return b_primal[:, self.truncate_idx[0]:self.truncate_idx[1]], b_primal

    def forward(self, b_primal):
        b, b_0 = self.truncate(b_primal)
        for i in range(0, len(self.optimality_layers) - 1, 2):
            b = F.relu(self.optimality_layers[i](b))
            b = self.optimality_layers[i+1](b)  # Apply LayerNorm
        out = self.optimality_layers[-1](b)

        return out, out, 0


class POCS(nn.Module):
    def __init__(self, free_idx, A, WzProj, max_iter, eq_tol, ineq_tol):
        super(POCS, self).__init__()

        self.free_num = free_idx[1] + 1
        self.A = A.requires_grad_(False)
        self.Weight_Proj = WzProj.t().requires_grad_(False)
        self.max_iter = max_iter
        self.eq_tol = eq_tol
        self.ineq_tol = ineq_tol

    def stopping_criterion(self, z, b_0):
        # gurobi uses the 1. worst constraint violation 2. scale the tolerance by D1 @ tolerance
        # but gurobi does not use the eq_mean / eq_scale_mean
        # batch mean to avoid black sheep in training samples
        with torch.no_grad():
            # eq_residual = z @ self.A.t() - b_0
            # eq_rhs = b_0
            # eq_mean = torch.norm(eq_residual, p=2, dim=-1).mean()
            # eq_scale_mean = 1 + torch.norm(eq_rhs, p=2, dim=-1).mean()
            # eq_stopping_criterion = eq_mean / eq_scale_mean
            # eq_stopping_criterion = torch.mean(torch.abs(eq_residual), dim=0)  # (bsz, const_num) -> (const_num,)

            ineq_residual = torch.relu(-z[:, self.free_num:])
            # ineq_stopping_criterion = torch.norm(ineq_residual, p=2, dim=-1).mean()
            ineq_stopping_criterion = torch.mean(torch.abs(ineq_residual), dim=0)  # (bsz, const_num) -> (const_num,)
            return ineq_stopping_criterion

    def forward(self, z, Bias_Proj, b_0):
        z = Bias_Proj + z @ self.Weight_Proj  # z0 \in set A
        curr_iter = 0
        while curr_iter <= self.max_iter:
            P2z = z.clone()
            P2z[:, self.free_num:] = F.relu(P2z[:, self.free_num:])
            z = Bias_Proj + P2z @ self.Weight_Proj
            curr_iter += 1
            ineq_stopping_criterion = self.stopping_criterion(z, b_0)
            if (ineq_stopping_criterion <= self.ineq_tol).all():
                break
        z_star = z
        return z_star, curr_iter


class EAPM(nn.Module):
    def __init__(self, free_idx, A, WzProj, max_iter, eq_tol, ineq_tol, rho=1.0):
        super(EAPM, self).__init__()

        self.free_num = free_idx[1] + 1
        self.A = A.requires_grad_(False)
        self.Weight_Proj = WzProj.t().requires_grad_(False)
        self.max_iter = max_iter
        self.eq_tol = eq_tol
        self.ineq_tol = ineq_tol
        self.rho = rho

    def stopping_criterion_old(self, z, b_0):
        # gurobi uses the 1. worst constraint violation 2. scale the tolerance by D1 @ tolerance
        # but gurobi does not use the eq_mean / eq_scale_mean
        # batch mean to avoid black sheep in training samples
        with torch.no_grad():
            # eq_residual = z @ self.A.t() - b_0
            # eq_stopping_criterion = torch.mean(torch.abs(eq_residual), dim=0)  # (bsz, const_num) -> (const_num,)

            ineq_residual = torch.relu(-z[:, self.free_num:])
            ineq_stopping_criterion = torch.mean(torch.abs(ineq_residual), dim=0)  # (bsz, const_num) -> (const_num,)
            return ineq_stopping_criterion

    def stopping_criterion(self, z, b_0):
        with torch.no_grad():
            eq_residual = z @ self.A.t() - b_0
            eq_lhs = torch.mean(torch.abs(eq_residual), dim=0)
            ineq_residual = torch.relu(-z[:, self.free_num:])
            ineq_lhs = torch.mean(torch.abs(ineq_residual), dim=0)
            lhs = torch.sqrt(eq_lhs.pow(2).sum() + ineq_lhs.pow(2).sum())
            rhs = 1 + torch.sqrt(torch.mean(b_0, dim=0).pow(2).sum() + 0)

            return lhs/rhs

    def forward(self, z, Bias_Proj, b_0):
        z = Bias_Proj + z @ self.Weight_Proj  # z0 \in set A

        # print('\n\nA NEW POINT PROJECTION STARTS')
        # eq_stopping_criterion = torch.mean(torch.abs(z @ self.A.t() - b_0), dim=0)
        # print(f'\n\nthe eq_stopping_criterion (before loop) is {eq_stopping_criterion}')
        # print(f'the number of eq violation {((eq_stopping_criterion <= self.eq_tol) == False).sum().item()}')
        #
        # z = Bias_Proj + z @ self.Weight_Proj  # z0 \in set A
        # eq_stopping_criterion = torch.mean(torch.abs(z @ self.A.t() - b_0), dim=0)
        # print(f'\n\nthe eq_stopping_criterion (before loop BUT 2ND) is {eq_stopping_criterion}')
        # print(f'the number of eq violation {((eq_stopping_criterion <= self.eq_tol) == False).sum().item()}')

        curr_iter = 0
        while curr_iter <= self.max_iter:
            P2z = z.clone()
            P2z[:, self.free_num:] = F.relu(P2z[:, self.free_num:])
            P1P2z = Bias_Proj + P2z @ self.Weight_Proj

            # print(f'\n\n=================')
            # print(f'In iteration {curr_iter}')
            # eq_stopping_criterion = torch.mean(torch.abs(P1P2z @ self.A.t() - b_0), dim=0)
            # print(f'the eq_stopping_criterion (before numerical issue) is {eq_stopping_criterion}')
            # print(f'the number of eq violation {((eq_stopping_criterion <= self.eq_tol)==False).sum().item()}')

            residual = P1P2z - z
            # compute the K
            mask = (z[:, self.free_num:] >= 0).all(dim=-1)
            K = torch.ones(residual.shape[0]).to(device)
            K[~mask] = (P2z[~mask] - z[~mask]).pow(2).sum(dim=-1) / residual[~mask].pow(2).sum(dim=-1)
            z = z + self.rho * K.unsqueeze(-1) * residual
            # avoid numerical issue
            z = Bias_Proj + z @ self.Weight_Proj

            # eq_stopping_criterion = torch.mean(torch.abs(z @ self.A.t() - b_0), dim=0)
            # print(f'the eq_stopping_criterion (AFTER numerical issue) is {eq_stopping_criterion}')
            # print(f'the number of eq violation {((eq_stopping_criterion <= self.eq_tol) == False).sum().item()}')

            curr_iter += 1
            violation = self.stopping_criterion(z, b_0)
            if violation <= self.ineq_tol:
                break

            # ineq_stopping_criterion = self.stopping_criterion(z, b_0)
            # print(f'the ineq_stopping_criterion is {ineq_stopping_criterion}')
            # print(f'the number of ineq violation {((ineq_stopping_criterion <= self.ineq_tol)==False).sum().item()}')
            # if (ineq_stopping_criterion <= self.ineq_tol).all():
            #     break
        z_star = z
        return z_star, curr_iter


class PeriodicEAPM(nn.Module):
    def __init__(self, free_idx, A, WzProj, max_iter, eq_tol, ineq_tol, rho=1.0):
        super(PeriodicEAPM, self).__init__()

        self.free_num = free_idx[1] + 1
        self.A = A.requires_grad_(False)
        self.Weight_Proj = WzProj.t().requires_grad_(False)
        self.max_iter = max_iter
        self.eq_tol = eq_tol
        self.ineq_tol = ineq_tol
        self.rho = rho

    def stopping_criterion(self, z, b_0):
        # gurobi uses the 1. worst constraint violation 2. scale the tolerance by D1 @ tolerance
        # but gurobi does not use the eq_mean / eq_scale_mean
        # batch mean to avoid black sheep in training samples
        with torch.no_grad():
            # eq_residual = z @ self.A.t() - b_0
            # eq_stopping_criterion = torch.mean(torch.abs(eq_residual), dim=0)  # (bsz, const_num) -> (const_num,)

            ineq_residual = torch.relu(-z[:, self.free_num:])
            ineq_stopping_criterion = torch.mean(torch.abs(ineq_residual), dim=0)  # (bsz, const_num) -> (const_num,)
            return ineq_stopping_criterion

    def forward(self, z, Bias_Proj, b_0):
        z = Bias_Proj + z @ self.Weight_Proj  # z0 \in set A
        curr_iter = 0
        while curr_iter <= self.max_iter:
            P2z = z.clone()
            P2z[:, self.free_num:] = F.relu(P2z[:, self.free_num:])
            P1P2z = Bias_Proj + P2z @ self.Weight_Proj
            residual = P1P2z - z
            # compute the K
            mask = (z[:, self.free_num:] >= 0).all(dim=-1)
            K = torch.ones(residual.shape[0]).to(device)
            K[~mask] = (P2z[~mask] - z[~mask]).pow(2).sum(dim=-1) / residual[~mask].pow(2).sum(dim=-1)
            # compute y_new (periodic centering)
            if curr_iter % 3 == 2:
                z = z + self.rho / 2 * K.unsqueeze(-1) * residual
            else:
                z = z + self.rho * K.unsqueeze(-1) * residual
            # # compute y_new
            # z = z + self.rho * K.unsqueeze(-1) * residual
            # avoid numerical issue
            z = Bias_Proj + z @ self.Weight_Proj
            curr_iter += 1
            ineq_stopping_criterion = self.stopping_criterion(z, b_0)
            if (ineq_stopping_criterion <= self.ineq_tol).all():
                break
        z_star = z
        return z_star, curr_iter


class NullSpace(nn.Module):
    """
    z := Bias_Proj + z @ self.Weight_Proj # now z in set A
    N: orthogonal basis of the null space of A (todo: Check the packages for this)
    see derivation in 草稿纸 in ipad: s_star = N^T ReLu(-z)
    s_star: the s that makes z_star = z + N s_star >= 0 (of course A z_star = b)
    Therefore, z_star = z + N N^T ReLu(-z)
    """
    def __init__(self, free_idx, A, N, WzProj, max_iter, eq_tol, ineq_tol, rho=1.0):
        super(NullSpace, self).__init__()

        self.free_num = free_idx[1] + 1
        self.A = A.requires_grad_(False)
        self.N = N.requires_grad_(False)
        self.Weight_Proj = WzProj.t().requires_grad_(False)
        self.max_iter = max_iter
        self.eq_tol = eq_tol
        self.ineq_tol = ineq_tol
        self.rho = rho

    def stopping_criterion(self, z, b_0):
        # gurobi uses the 1. worst constraint violation 2. scale the tolerance by D1 @ tolerance
        # but gurobi does not use the eq_mean / eq_scale_mean
        # batch mean to avoid black sheep in training samples
        with torch.no_grad():
            ineq_residual = torch.relu(-z[:, self.free_num:])
            ineq_stopping_criterion = torch.mean(torch.abs(ineq_residual), dim=0)  # (bsz, const_num) -> (const_num,)
            return ineq_stopping_criterion

    def forward(self, z, Bias_Proj, b_0):
        z = Bias_Proj + z @ self.Weight_Proj  # z0 \in set A
        z = z + -z @ self.N @ self.N.t()
        curr_iter = 0
        while curr_iter <= self.max_iter:
            P2z = z.clone()
            P2z[:, self.free_num:] = F.relu(P2z[:, self.free_num:])
            z = Bias_Proj + P2z @ self.Weight_Proj
            z = z + -z @ self.N @ self.N.t()
            curr_iter += 1
            ineq_stopping_criterion = self.stopping_criterion(z, b_0)
            if (ineq_stopping_criterion <= self.ineq_tol).all():
                break
        z_star = z
        return z_star, curr_iter






        # # todo: integrate nullspace into eapm
        # z = Bias_Proj + z @ self.Weight_Proj  # z0 \in set A
        #
        # # null space block
        # # z_f, z_nf = z[:, :self.free_num], z[:, self.free_num:]
        # # s = -z_nf @ self.N
        # # z_nf = z_nf + s @ self.N.t()
        # # z = torch.cat((z_f, z_nf), dim=-1)
        # z = z + -z @ self.N @ self.N.t()
        # # null space block
        #
        # curr_iter = 0
        # while curr_iter <= self.max_iter:
        #     P2z = z.clone()
        #     P2z[:, self.free_num:] = F.relu(P2z[:, self.free_num:])
        #     P1P2z = Bias_Proj + P2z @ self.Weight_Proj
        #
        #     # null space block
        #     # P1P2z_f, P1P2z_nf = P1P2z[:, :self.free_num], P1P2z[:, self.free_num:]
        #     # s = -P1P2z_nf @ self.N
        #     # P1P2z_nf = P1P2z_nf + s @ self.N.t()
        #     # P1P2z = torch.cat((P1P2z_f, P1P2z_nf), dim=-1)
        #     P1P2z = P1P2z + -P1P2z @ self.N @ self.N.t()
        #     # null space block
        #
        #     residual = P1P2z - z
        #     # compute the K
        #     mask = (z[:, self.free_num:] >= 0).all(dim=-1)
        #     K = torch.ones(residual.shape[0]).to(device)
        #     K[~mask] = (P2z[~mask] - z[~mask]).pow(2).sum(dim=-1) / residual[~mask].pow(2).sum(dim=-1)
        #     # compute y_new (periodic centering)
        #     # if curr_iter % 3 == 2:
        #     #     z = z + self.rho / 2 * K.unsqueeze(-1) * residual
        #     # else:
        #     #     z = z + self.rho * K.unsqueeze(-1) * residual
        #     # compute y_new
        #     z = z + self.rho * K.unsqueeze(-1) * residual
        #     # avoid numerical issue
        #     z = Bias_Proj + z @ self.Weight_Proj
        #     curr_iter += 1
        #     ineq_stopping_criterion = self.stopping_criterion(z, b_0)
        #     if (ineq_stopping_criterion <= self.ineq_tol).all():
        #         break
        # z_star = z
        # return z_star, curr_iter


class OptProjNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, output_dim,
                 truncate_idx, free_idx, A, N, WzProj, WbProj,
                 max_iter, eq_tol, ineq_tol, proj_method='POCS', rho=1.0):
        super(OptProjNN, self).__init__()

        self.optimality_layers = OptimalityLayers(input_dim, hidden_dim, hidden_num, output_dim, truncate_idx)
        self.init_projection(free_idx, A, N, WzProj, max_iter, eq_tol, ineq_tol, proj_method, rho)
        self.WbProj = WbProj.requires_grad_(False)

    def init_projection(self, free_idx, A, N, WzProj, max_iter, eq_tol, ineq_tol, proj_method, rho):
        if proj_method == 'POCS':
            self.projection = POCS(free_idx, A, WzProj, max_iter, eq_tol, ineq_tol)
        elif proj_method == 'EAPM':
            self.projection = EAPM(free_idx, A, WzProj, max_iter, eq_tol, ineq_tol, rho)
        elif proj_method == 'PeriodicEAPM':
            # todo: the code in may doesn't have this line. Make sure the code in PeriodicEAPM is correct
            self.projection = PeriodicEAPM(free_idx, A, WzProj, max_iter, eq_tol, ineq_tol, rho)
        elif proj_method == 'NullSpace':
            self.projection = NullSpace(free_idx, A, N, WzProj, max_iter, eq_tol, ineq_tol)
        else:
            raise ValueError('Invalid projection method')

    def forward(self, b_primal):
        z, _, _ = self.optimality_layers(b_primal)
        with torch.no_grad():
            Bias_Proj = b_primal @ self.WbProj.t()
        z_star, proj_num = self.projection(z, Bias_Proj, b_primal)
        return z_star, z, proj_num


class DualOptProjNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, output_dim,
                 truncate_idx, free_idx, A, WzProj, WbProj,
                 max_iter, eq_tol, ineq_tol, projection='POCS', rho=1.0, b_dual=None):
        super(DualOptProjNN, self).__init__()

        self.optimality_layers = OptimalityLayers(input_dim, hidden_dim, hidden_num, output_dim, truncate_idx)
        self.init_projection(free_idx, A, WzProj, max_iter, eq_tol, ineq_tol, projection, rho)
        self.Bias_Proj = (b_dual @ WbProj.t()).requires_grad_(False)

    def init_projection(self, free_idx, A, WzProj, max_iter, eq_tol, ineq_tol, projection, rho):
        if projection == 'POCS':
            self.projection = POCS(free_idx, A, WzProj, max_iter, eq_tol, ineq_tol)
        elif projection == 'EAPM':
            self.projection = EAPM(free_idx, A, WzProj, max_iter, eq_tol, ineq_tol, rho)
        else:
            raise ValueError('Invalid projection method')

    def forward(self, b_primal):
        z, _, _ = self.optimality_layers(b_primal)
        z_star, proj_num = self.projection(z, self.Bias_Proj, b_primal)
        return z_star, z, proj_num





class VanillaNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, output_dim,
                 truncate_idx):
        super(VanillaNN, self).__init__()

        self.truncate_idx = truncate_idx
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(hidden_num - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def initialize_final_layer(self):
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, b_primal):
        b_primal = b_primal[:, self.truncate_idx[0]:self.truncate_idx[1]]
        for layer in self.layers[:-1]:
            b_primal = F.relu(layer(b_primal))
        out = self.layers[-1](b_primal)

        if self.training:
            return out, out, 0
        else:
            return out, out, 0


