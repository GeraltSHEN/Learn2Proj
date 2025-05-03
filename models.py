import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.nonnegative_mask = nonnegative_mask
        self.eq_projector = Projector(weight=eq_weight, bias_transform=eq_bias_transform)

    def forward(self, x):
        x = torch.where(self.nonnegative_mask, torch.clamp(x, min=0), x)
        x_eq = self.eq_projector(x)
        return x_eq


class LDRPM(nn.Module):
    def __init__(self, nonnegative_mask, eq_weight, eq_bias_transform, ldr_weight, ldr_bias):
        super(LDRPM, self).__init__()
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


class FeasibilityNet:
    def __init__(self, algo, eq_tol, ineq_tol, max_iters, **kwargs):
        super(FeasibilityNet, self).__init__()

        self.algo = algo
        self.eq_tol = eq_tol
        self.ineq_tol = ineq_tol
        self.max_iters = max_iters

        self.eq_epsilon = None
        self.ineq_epsilon = None
        self.iters = 0

    def __call__(self, x, A_transpose, b, nonnegative_mask):
        self.iters = 0

        self.eq_epsilon, self.ineq_epsilon = self.stopping_criterion(x, A_transpose, b, nonnegative_mask)
        while ((self.eq_epsilon.mean() > self.eq_tol or self.ineq_epsilon.mean() > self.ineq_tol)
               and self.iters < self.max_iters):
            x = self.algo(x)
            self.iters += 1

        return x

    @staticmethod
    def stopping_criterion(x, A_transpose, b, nonnegative_mask):
        with torch.no_grad():
            eq_residual = x @ A_transpose - b
            ineq_residual = torch.relu(-x[nonnegative_mask])

            eq_violation = torch.norm(eq_residual, p=2, dim=-1)
            ineq_violation = torch.norm(ineq_residual, p=2, dim=-1)

            eq_epsilon = eq_violation / (1 + torch.norm(b, p=2, dim=-1))  # scaled by the norm of b
            ineq_epsilon = ineq_violation  # no scaling because rhs is 0
            return eq_epsilon, ineq_epsilon


# class CombinedModel(nn.Module):
#     def __init__(self, optimality_net, feasibility_net):
#         super().__init__()
#         self.optimality_net = optimality_net
#         self.feasibility_net = feasibility_net
#
#     def forward(self, features):
#         x = self.optimality_net(features)
#         x = self.feasibility_net(x)
#         return x










