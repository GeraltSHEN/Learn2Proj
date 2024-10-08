import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class OptimalityLayers(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, output_dim,
                 mutable_idx):
        super(OptimalityLayers, self).__init__()

        self.mutable_idx = mutable_idx

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

    def xavier_init(self):
        for layer in self.optimality_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, b_primal):
        b = b_primal[:, self.mutable_idx]
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
        curr_iter = 0
        while curr_iter <= self.max_iter:
            P2z = z.clone()
            P2z[:, self.free_num:] = F.relu(P2z[:, self.free_num:])
            z = Bias_Proj + P2z @ self.Weight_Proj

            curr_iter += 1
            violation = self.stopping_criterion(z, b_0)
            if violation <= self.ineq_tol:
                break
        z_star = z
        return z_star, curr_iter, torch.zeros(z.shape[0]).to(device)


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
        return z_star, curr_iter, torch.zeros(z.shape[0]).to(device)


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
            # avoid numerical issue
            z = Bias_Proj + z @ self.Weight_Proj

            curr_iter += 1
            violation = self.stopping_criterion(z, b_0)
            if violation <= self.ineq_tol:
                break
        z_star = z
        return z_star, curr_iter, torch.zeros(z.shape[0]).to(device)


class LDRPM(nn.Module):
    def __init__(self, mutable_idx, free_idx, A, WzProj, Q, z0, eq_tol, ineq_tol):
        super(LDRPM, self).__init__()

        self.mutable_idx = mutable_idx
        self.free_num = free_idx[1] + 1
        self.A = A.requires_grad_(False)
        self.Weight_Proj = WzProj.t().requires_grad_(False)
        self.Q = Q.t().requires_grad_(False)
        self.z0 = z0.requires_grad_(False)
        self.eq_tol = eq_tol
        self.ineq_tol = ineq_tol

    def stopping_criterion(self, z, b_0):
        with torch.no_grad():
            eq_residual = z @ self.A.t() - b_0
            eq_lhs = torch.mean(torch.abs(eq_residual), dim=0)
            ineq_residual = torch.relu(-z[:, self.free_num:])
            ineq_lhs = torch.mean(torch.abs(ineq_residual), dim=0)
            lhs = torch.sqrt(eq_lhs.pow(2).sum() + ineq_lhs.pow(2).sum())
            rhs = 1 + torch.sqrt(torch.mean(b_0, dim=0).pow(2).sum() + 0)

            return lhs / rhs

    def forward(self, z, Bias_Proj, b_0):
        b_truncated = b_0[:, self.mutable_idx]
        z_LDR = self.z0 + b_truncated @ self.Q  # (bsz, const_num) @ (const_num, var_num) -> (bsz, var_num)
        z_eq = Bias_Proj + z @ self.Weight_Proj  # z0 \in set A

        mask_violation = z_eq < 0

        masked_z_eq = z_eq * mask_violation
        masked_s = z_LDR - masked_z_eq
        # disturb the 0s in masked_s to avoid division by 0
        # 0s in masked_s only caused by z_LDR_i = 0 and masked_z_eq_i = 0
        # disturbance doesn't affect the result as the alpha for these indexes will be 0 (masked_z_eq_i = 0)
        masked_s[masked_s == 0] = 99

        alphas = - masked_z_eq / masked_s  # (bsz, var_num)
        alpha = torch.max(alphas, dim=1).values  # (bsz, )
        # if torch.isnan(alphas).any():
        #     print('alphas has nan')
        #     print(f'index of nan is {torch.isnan(alphas).nonzero(as_tuple=True)}')
        #     for idx in torch.isnan(alphas).nonzero(as_tuple=True)[1].tolist():
        #         print(f'nan idx is {idx}')
        #         print(f'z_LDR values at this idx: {z_LDR[:, idx - 2:idx + 2]}')
        #         print(f'masked_z_eq values at this idx: {masked_z_eq[:, idx - 2:idx + 2]}')
        #         print(f'mask_violation values at this idx: {mask_violation[:, idx - 2:idx + 2]}')
        #         print(f's values at this idx: {s[:, idx - 2:idx + 2]}')
        #         print(f'alphas values at this idx: {alphas[:, idx - 2:idx + 2]}')

        z_star = z_LDR * alpha.unsqueeze(1) + z_eq * (1 - alpha).unsqueeze(1)
        return z_star, 0, alpha


class GMDS(nn.Module):
    def __init__(self, free_idx, A_eq_inp, A_eq_dep, A_ineq_inp, A_ineq_dep,
                 b_eq, b_ineq, z_int,
                 eq_tol, ineq_tol):
        super(GMDS, self).__init__()

        self.free_num = free_idx[1] + 1

        self.A_eq_inp = A_eq_inp.requires_grad_(False)
        self.A_eq_dep = A_eq_dep.requires_grad_(False)
        self.A_ineq_inp = A_ineq_inp.requires_grad_(False)
        self.A_ineq_dep = A_ineq_dep.requires_grad_(False)
        # eq (7) in paper; remember this is for S_ref not S_ref_bar
        self.A = A_ineq_inp - A_ineq_dep @ torch.inverse(A_eq_dep) @ A_eq_inp
        self.b = b_ineq - A_ineq_dep @ torch.inverse(A_eq_dep) @ b_eq
        self.A_bar = self.A
        self.b_bar = self.b + z_int

        self.eq_tol = eq_tol
        self.ineq_tol = ineq_tol

        self.inp_num = A_eq_inp.shape[1]
        self.H_unitball = torch.cat([torch.eye(self.inp_num), -torch.eye(self.inp_num)], dim=0).to(device)
        self.h_unitball = torch.ones(2 * self.inp_num).to(device)

    def stopping_criterion(self, z, b_0):
        with torch.no_grad():
            eq_residual = z @ self.A.t() - b_0
            eq_lhs = torch.mean(torch.abs(eq_residual), dim=0)
            ineq_residual = torch.relu(-z[:, self.free_num:])
            ineq_lhs = torch.mean(torch.abs(ineq_residual), dim=0)
            lhs = torch.sqrt(eq_lhs.pow(2).sum() + ineq_lhs.pow(2).sum())
            rhs = 1 + torch.sqrt(torch.mean(b_0, dim=0).pow(2).sum() + 0)

            return lhs/rhs

    def gm_unitball(self, v):
        # eq (9) in paper
        lhs = (v @ self.H_unitball.t())  # (bsz, 2 * inp_num)
        ratio = lhs / self.h_unitball  # (bsz, 2 * inp_num) actually you don't need h_unitball because they are all 1s
        gm = torch.max(ratio, dim=1).values  # (bsz, )
        return gm

    def gm_Sbar(self, v):
        # eq (9) in paper
        lhs = (v @ self.A_bar.t())  # (bsz, const_num)
        ratio = lhs / self.b_bar  # (bsz, const_num)
        gm = torch.max(ratio, dim=1).values  # (bsz, )
        return gm

    def eq_completion(self, z_inp_star):
        # eq (6) in paper
        #todo: b_0 should be b_eq check!!!!
        return torch.inverse(self.A_eq_dep) @ self.A_eq_inp @ z_inp_star - torch.inverse(self.A_eq_dep) @ self.b_eq

    def forward(self, v, z_int, b_0):
        """
notations:
Paper: u ---> z
Paper: x ---> truncated b (the changing parameters)
Paper: b ---> b (the constant parameters)
        """
        v_star = torch.clamp(v, min=-1, max=1)  # put it in L-inf unit ball
        z_inp_star = self.gm_unitball(v_star) / self.gm_Sbar(v_star) * v_star + z_int  # eq (11) in paper
        z_dep_star = self.eq_completion(z_inp_star, b_0)  # todo: implement eq (6) in paper
        z_star = torch.cat([z_inp_star, z_dep_star], dim=1)
        return z_star, 0, torch.zeros(z.shape[0]).to(device)




class OptProjNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, output_dim,
                 mutable_idx, free_idx, A, Q, z0, WzProj, WbProj,
                 max_iter, eq_tol, ineq_tol, proj_method='POCS', rho=1.0):
        super(OptProjNN, self).__init__()

        self.optimality_layers = OptimalityLayers(input_dim, hidden_dim, hidden_num, output_dim, mutable_idx)
        self.init_projection(mutable_idx, free_idx, A, Q, z0, WzProj, max_iter, eq_tol, ineq_tol, proj_method, rho)
        self.WbProj = WbProj.requires_grad_(False)

    def init_projection(self, mutable_idx, free_idx, A, Q, z0, WzProj, max_iter, eq_tol, ineq_tol, proj_method, rho):
        if proj_method == 'POCS':
            self.projection = POCS(free_idx, A, WzProj, max_iter, eq_tol, ineq_tol)
        elif proj_method == 'EAPM':
            self.projection = EAPM(free_idx, A, WzProj, max_iter, eq_tol, ineq_tol, rho)
        elif proj_method == 'PeriodicEAPM':
            self.projection = PeriodicEAPM(free_idx, A, WzProj, max_iter, eq_tol, ineq_tol, rho)
        elif proj_method == 'LDRPM':
            self.projection = LDRPM(mutable_idx, free_idx, A, WzProj, Q, z0, eq_tol, ineq_tol)
        else:
            raise ValueError('Invalid projection method')

    def forward(self, b_primal):
        z, _, _ = self.optimality_layers(b_primal)
        with torch.no_grad():
            Bias_Proj = b_primal @ self.WbProj.t()
        z_star, proj_num, alpha = self.projection(z, Bias_Proj, b_primal)
        return z_star, z, proj_num, alpha





class LDRNNSplit(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, output_dim,
                 mutable_idx, free_idx, A, Q, z0, WzProj, WbProj,
                 max_iter, eq_tol, ineq_tol):
        super(LDRNNSplit, self).__init__()

        self.mutable_idx = mutable_idx

        # optimality layers
        self.optimality_layers = OptimalityLayers(input_dim, hidden_dim, hidden_num, output_dim, mutable_idx)

        self.free_num = free_idx[1] + 1
        self.A = A.requires_grad_(False)
        self.Weight_Proj = WzProj.t().requires_grad_(False)
        self.WbProj = WbProj.requires_grad_(False)
        self.Q = Q.t().requires_grad_(False)
        self.z0 = z0.requires_grad_(False)
        self.eq_tol = eq_tol
        self.ineq_tol = ineq_tol


    def stopping_criterion(self, z, b_0):
        with torch.no_grad():
            eq_residual = z @ self.A.t() - b_0
            eq_lhs = torch.mean(torch.abs(eq_residual), dim=0)
            ineq_residual = torch.relu(-z[:, self.free_num:])
            ineq_lhs = torch.mean(torch.abs(ineq_residual), dim=0)
            lhs = torch.sqrt(eq_lhs.pow(2).sum() + ineq_lhs.pow(2).sum())
            rhs = 1 + torch.sqrt(torch.mean(b_0, dim=0).pow(2).sum() + 0)

            return lhs / rhs

    def forward(self, b_primal):
        z, _, _ = self.optimality_layers(b_primal)
        with torch.no_grad():
            Bias_Proj = b_primal @ self.WbProj.t()

        b_truncated = b_primal[:, self.mutable_idx]
        z_LDR = self.z0 + b_truncated @ self.Q  # (bsz, const_num) @ (const_num, var_num) -> (bsz, var_num)
        z_eq = Bias_Proj + z @ self.Weight_Proj  # z0 \in set A

        mask_violation = z_eq < 0

        masked_z_eq = z_eq * mask_violation
        masked_s = z_LDR - masked_z_eq
        # disturb the 0s in masked_s to avoid division by 0
        # 0s in masked_s only caused by z_LDR_i = 0 and masked_z_eq_i = 0
        # disturbance doesn't affect the result as the alpha for these indexes will be 0 (masked_z_eq_i = 0)
        masked_s[masked_s == 0] = 99

        alphas = - masked_z_eq / masked_s  # (bsz, var_num)
        alpha = torch.max(alphas, dim=1).values  # (bsz, )

        if self.training:
            return z_LDR, z_eq, 0, alpha
        else:
            z_star = z_LDR * alpha.unsqueeze(1) + z_eq * (1 - alpha).unsqueeze(1)
            return z_star, z, 0, alpha












# todo: this must be updated, i guess bugs must happen
class DualOptProjNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, output_dim,
                 mutable_idx, free_idx, A, WzProj, WbProj,
                 max_iter, eq_tol, ineq_tol, projection='POCS', rho=1.0, b_dual=None):
        super(DualOptProjNN, self).__init__()

        self.optimality_layers = OptimalityLayers(input_dim, hidden_dim, hidden_num, output_dim, mutable_idx)
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
                 mutable_idx):
        super(VanillaNN, self).__init__()

        self.mutable_idx = mutable_idx
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(hidden_num - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def initialize_final_layer(self):
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, b_primal):
        b_primal = b_primal[:, self.mutable_idx]
        for layer in self.layers[:-1]:
            b_primal = F.relu(layer(b_primal))
        out = self.layers[-1](b_primal)

        if self.training:
            return out, out, 0, torch.zeros(out.shape[0]).to(device)
        else:
            return out, out, 0, torch.zeros(out.shape[0]).to(device)


