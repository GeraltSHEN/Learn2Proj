import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)


class PrimalCostPenalty(nn.Module):
    def __init__(self, c_primal, A_primal, primal_x_dim, primal_free_num, penalty_g, penalty_h):
        super(PrimalCostPenalty, self).__init__()
        self.c_primal = c_primal
        self.A_primal = A_primal

        self.primal_x_dim = primal_x_dim
        self.primal_free_num = primal_free_num

        self.penalty_g = penalty_g
        self.penalty_h = penalty_h

    def forward(self, x_primal, b_primal):
        f_cost = torch.matmul(x_primal, self.c_primal)
        u, v = torch.split(x_primal, [self.primal_free_num, self.primal_x_dim - self.primal_free_num], dim=-1)
        v = torch.relu(-v)
        g_cost = v.pow(2).sum(dim=-1)
        h_cost = (torch.matmul(x_primal, self.A_primal.t()) - b_primal).pow(2).sum(dim=-1)
        return (f_cost + self.penalty_g * g_cost + self.penalty_h * h_cost).mean()


class DualCost(nn.Module):
    def __init__(self):
        super(DualCost, self).__init__()

    def forward(self, y_dual, b_primal):
        """
        output: y_dual, e.g. (64, 191)
        target: dual cost, e.g. (64,)
        in_val: b_primal = - c_dual, e.g. (64, 191). NOTE c_dual is in max c_dual^T y_dual
        """
        return (torch.sum(y_dual * b_primal, dim=-1)).mean()


class DualCostPenalty(nn.Module):
    def __init__(self, b_dual, A_dual, dual_y_dim, dual_free_num, penalty_g, penalty_h):
        super(DualCostPenalty, self).__init__()
        self.b_dual = b_dual
        self.A_dual = A_dual

        self.dual_y_dim = dual_y_dim
        self.dual_free_num = dual_free_num

        self.penalty_g = penalty_g
        self.penalty_h = penalty_h

    def forward(self, y_dual, b_primal):
        f_cost = torch.sum(y_dual * b_primal, dim=-1)
        u, v = torch.split(y_dual, [self.dual_free_num, self.dual_y_dim - self.dual_free_num], dim=-1)
        v = torch.relu(-v)
        g_cost = v.pow(2).sum(dim=-1)
        h_cost = (torch.matmul(y_dual, self.A_dual.t()) - self.b_dual).pow(2).sum(dim=-1)
        return (f_cost + self.penalty_g * g_cost + self.penalty_h * h_cost).mean()


class PrimalQuadraticCost(nn.Module):
    def __init__(self, c_primal, Q_primal, cvx=True):
        super(PrimalQuadraticCost, self).__init__()
        self.c_primal = c_primal
        self.Q_primal = Q_primal
        self.cvx = cvx

    def forward(self, x_primal):
        #return (torch.sum(1 / 2 * torch.matmul(x_primal, self.Q_primal) * x_primal, dim=-1) + torch.matmul(x_primal, self.c_primal)).mean()
        quadratic_cost = torch.sum(1/2 * torch.matmul(x_primal, self.Q_primal) * x_primal, dim=-1)
        if self.cvx:
            linear_cost = torch.matmul(x_primal, self.c_primal)
        else:
            linear_cost = torch.matmul(torch.sin(x_primal), self.c_primal)
        return (quadratic_cost + linear_cost).mean()


class PrimalQuadraticCostPenalty(nn.Module):
    def __init__(self, c_primal, Q_primal, A_primal, primal_x_dim, primal_free_num, penalty_g, penalty_h, cvx=True):
        super(PrimalQuadraticCostPenalty, self).__init__()
        self.c_primal = c_primal
        self.Q_primal = Q_primal
        self.A_primal = A_primal

        self.primal_x_dim = primal_x_dim
        self.primal_free_num = primal_free_num

        self.penalty_g = penalty_g
        self.penalty_h = penalty_h

        self.cvx = cvx

    def forward(self, x_primal, b_primal):
        #f_cost = torch.sum(1 / 2 * torch.matmul(x_primal, self.Q_primal) * x_primal, dim=-1) + torch.matmul(x_primal, self.c_primal)
        f_cost = torch.sum(1/2 * torch.matmul(x_primal, self.Q_primal) * x_primal, dim=-1)
        if self.cvx:
            f_cost = f_cost + torch.matmul(x_primal, self.c_primal)
        else:
            f_cost = f_cost + torch.matmul(torch.sin(x_primal), self.c_primal)
        u, v = torch.split(x_primal, [self.primal_free_num, self.primal_x_dim - self.primal_free_num], dim=-1)
        v = torch.relu(-v)
        g_cost = v.pow(2).sum(dim=-1)
        h_cost = (torch.matmul(x_primal, self.A_primal.t()) - b_primal).pow(2).sum(dim=-1)
        return (f_cost + self.penalty_g * g_cost + self.penalty_h * h_cost).mean()


class RegLoss(nn.Module):
    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, z_star, z1):
        return F.mse_loss(z_star, z1)


def PrimalLoss(x_primal, c_primal, primal_cost_true):
    # primal_cost = PrimalCost(c_primal)
    # return primal_cost(x_primal) - primal_cost_true.mean()
    primal_cost = torch.matmul(x_primal, c_primal)
    optimality_gap = torch.abs((primal_cost - primal_cost_true) / primal_cost_true)
    return optimality_gap.mean()


def DualLoss(y_dual, b_primal, dual_cost_true):
    dual_cost = torch.sum(y_dual * b_primal, dim=-1)
    optimality_gap = torch.abs((dual_cost + dual_cost_true) / dual_cost_true)
    return optimality_gap.mean()
    #dual_cost = DualCost()
    #return dual_cost(y_dual, b_primal) + dual_cost_true.mean()


def PrimalQuadraticLoss(x_primal, c_primal, Q_primal, primal_cost_true, cvx=True):
    #primal_cost = torch.sum(1 / 2 * torch.matmul(x_primal, Q_primal) * x_primal, dim=-1) + torch.matmul(x_primal, c_primal)
    primal_cost = torch.sum(1 / 2 * torch.matmul(x_primal, Q_primal) * x_primal, dim=-1)
    if cvx:
        primal_cost = primal_cost + torch.matmul(x_primal, c_primal)
    else:
        primal_cost = primal_cost + torch.matmul(torch.sin(x_primal), c_primal)
    optimality_gap = torch.abs((primal_cost - primal_cost_true) / primal_cost_true)
    return optimality_gap.mean()


def StrongDualityGap(x_primal, c_primal, y_dual, b_primal):
    primal_cost = PrimalCost(c_primal)
    dual_cost = DualCost()
    return primal_cost(x_primal) + dual_cost(y_dual, b_primal)


def AugLagrangian(x_primal, c_primal, mu_dual_k, lambda_dual_k, b_primal, A_primal, rho, primal_free_num):
    f = torch.matmul(x_primal, c_primal)  #todo: not mean
    # Note: minus sign!!!!
    g = - x_primal[:, primal_free_num:]
    h = torch.matmul(x_primal, A_primal.t()) - b_primal
    # mu_g = (torch.sum(mu_dual_k * g, dim=-1)).mean()
    # lambda_h = (torch.sum(lambda_dual_k * h, dim=-1)).mean()
    # nu_g = (F.relu(g)).pow(2).sum()
    # nu_h = h.pow(2).sum()
    # return primal_cost(x_primal) + mu_g + lambda_h + rho / 2 * (nu_g + nu_h)
    mu_g = torch.sum(mu_dual_k * g, dim=-1)
    lambda_h = torch.sum(lambda_dual_k * h, dim=-1)
    nu_g = F.relu(g).pow(2).sum(dim=-1)
    nu_h = h.pow(2).sum(dim=-1)
    return (f + mu_g + lambda_h + rho / 2 * (nu_g + nu_h)).mean()


def QuadraticAugLagrangian(x_primal, Q_primal, c_primal, mu_dual_k, lambda_dual_k, b_primal, A_primal, rho, primal_free_num):
    f = torch.sum(1/2 * torch.matmul(x_primal, Q_primal) * x_primal, dim=-1) + torch.matmul(x_primal, c_primal)
    # Note: minus sign!!!!
    g = - x_primal[:, primal_free_num:]
    h = torch.matmul(x_primal, A_primal.t()) - b_primal
    mu_g = torch.sum(mu_dual_k * g, dim=-1)
    lambda_h = torch.sum(lambda_dual_k * h, dim=-1)
    nu_g = F.relu(g).pow(2).sum(dim=-1)
    nu_h = h.pow(2).sum(dim=-1)
    return (f + mu_g + lambda_h + rho / 2 * (nu_g + nu_h)).mean()


def ProximalLoss(x_primal, mu_dual, lambda_dual, mu_dual_k, lambda_dual_k, b_primal, A_primal, rho,
                 primal_free_num):
    g = - x_primal[:, primal_free_num:]
    # if device == 'cuda':
    #     g = torch.sparse.mm(x_primal, primal_exclude_free.t())
    # else:
    #     g = torch.matmul(x_primal, primal_exclude_free.t())
    h = torch.matmul(x_primal, A_primal.t()) - b_primal
    mu_dual_true = F.relu(mu_dual_k + rho * g)
    lambda_dual_true = lambda_dual_k + rho * h
    return F.mse_loss(mu_dual, mu_dual_true) + F.mse_loss(lambda_dual, lambda_dual_true)


def ProjAugLagrangian(z_star, z1, lambda_k, rho):
    # it doesn't contain the cost term (because it can either be primal or dual cost)
    h = z_star - z1
    lambda_h = (torch.sum(lambda_k * h, dim=-1)).mean()
    nu_h = h.pow(2).sum()
    return lambda_h + rho / 2 * nu_h


def ProjProximalLoss(z_star_k, z1_k, lambda_new, lambda_k, rho):
    h = z_star_k - z1_k
    lambda_dual_true = lambda_k + rho * h
    return F.mse_loss(lambda_new, lambda_dual_true)
