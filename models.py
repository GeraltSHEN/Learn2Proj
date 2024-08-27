import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

#import torch.autograd.Function as Function

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)


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
            # if curr_iter % 3 == 2:
            #     z = z + self.rho / 2 * K.unsqueeze(-1) * residual
            # else:
            #     z = z + self.rho * K.unsqueeze(-1) * residual
            # compute y_new
            z = z + self.rho * K.unsqueeze(-1) * residual
            # avoid numerical issue
            z = Bias_Proj + z @ self.Weight_Proj
            curr_iter += 1
            ineq_stopping_criterion = self.stopping_criterion(z, b_0)
            if (ineq_stopping_criterion <= self.ineq_tol).all():
                break
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


class OptProjNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, output_dim,
                 truncate_idx, free_idx, A, WzProj, WbProj,
                 max_iter, eq_tol, ineq_tol, projection='POCS', rho=1.0):
        super(OptProjNN, self).__init__()

        self.optimality_layers = OptimalityLayers(input_dim, hidden_dim, hidden_num, output_dim, truncate_idx)
        self.init_projection(free_idx, A, WzProj, max_iter, eq_tol, ineq_tol, projection, rho)
        self.WbProj = WbProj.requires_grad_(False)

    def init_projection(self, free_idx, A, WzProj, max_iter, eq_tol, ineq_tol, projection, rho):
        if projection == 'POCS':
            self.projection = POCS(free_idx, A, WzProj, max_iter, eq_tol, ineq_tol)
        elif projection == 'EAPM':
            self.projection = EAPM(free_idx, A, WzProj, max_iter, eq_tol, ineq_tol, rho)
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



        # if self.training:
        #     projections = ProjectionIter.apply
        #     z_star = projections(out, self.A, self.WzProj, self.WbProj, b_0,
        #                          self.max_iter, self.eq_tol, self.output_dim, self.free_num)
        #     return z_star, out
        #
        # else:
        #     with torch.no_grad():
        #         z = out
        #         curr_iter = 1
        #
        #         Bias_Proj = torch.matmul(b_0, self.WbProj.t())
        #         while curr_iter <= self.max_iter:
        #             z = Bias_Proj + torch.matmul(z, self.WzProj.t())
        #
        #             u, v = torch.split(z, [self.free_num, self.output_dim - self.free_num], dim=-1)
        #             v = F.relu(v)
        #             z = torch.cat((u, v), dim=-1)
        #
        #             curr_iter = curr_iter + 1
        #             eq_residual = z @ self.A.t() - b_0
        #             # worst_of_worst = torch.norm(eq_residual, float('inf')).item()
        #             # if worst_of_worst <= self.eq_tol:
        #             #     break
        #             eq_rhs = b_0  # another stopping criterion
        #             eq_mean = torch.norm(eq_residual, p=2, dim=-1).mean()  # another stopping criterion
        #             eq_scale_mean = 1 + torch.norm(eq_rhs, p=2, dim=-1).mean()  # another stopping criterion
        #             stopping_criterion = eq_mean / eq_scale_mean  # another stopping criterion
        #             if stopping_criterion <= self.eq_tol:  # another stopping criterion
        #                 break  # another stopping criterion
        #
        #         z_star = z
        #
        #         if self.report_projection:
        #             return z_star, out, curr_iter
        #         else:
        #             return z_star, out



        # # feasibility layers
        # self.projection = None  # POCS or EAPM
        #
        #
        # self.A = A.requires_grad_(False)
        # self.WzProj = WzProj.requires_grad_(False)
        # self.WbProj = WbProj.requires_grad_(False)
        #
        # self.max_iter = max_iter
        # self.eq_tol = eq_tol






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


