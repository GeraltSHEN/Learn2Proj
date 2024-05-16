import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)


class ProjectionIter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out, A, WzProj, WbProj, b, max_iter, f_tol, output_dim, free_num):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # Compute z_star from out and Compute J_accum simultaneously
        # Store any variables needed for backward in ctx

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        z = out
        curr_iter = 1
        bsz = z.shape[0]
        J_accum = torch.eye(output_dim).unsqueeze(0).repeat(bsz, 1, 1).requires_grad_(False).to(device)
        Bias_Proj = torch.matmul(b, WbProj.t())

        # start_time = time.time()  ####
        with torch.no_grad():
            while curr_iter <= max_iter:
                z = Bias_Proj + torch.matmul(z, WzProj.t())

                J_iter = WzProj.unsqueeze(0).repeat(bsz, 1, 1).requires_grad_(False).to(device)  # J^T actually
                u, v = torch.split(z, [free_num, output_dim - free_num], dim=-1)

                ref = z < 0
                ref[:, :free_num] = False
                J_iter[ref] = 0

                if device == 'cuda':
                    torch.cuda.synchronize()
                J_accum = torch.bmm(J_iter, J_accum)

                v = F.relu(v)
                z = torch.cat((u, v), dim=-1)

                curr_iter = curr_iter + 1
                eq_residual = z @ A.t() - b
                # worst_of_worst = torch.norm(eq_residual, float('inf')).item()
                # if worst_of_worst <= f_tol:
                #     break
                eq_rhs = b  # another stopping criterion
                eq_mean = torch.norm(eq_residual, p=2, dim=-1).mean()  # another stopping criterion
                eq_scale_mean = 1 + torch.norm(eq_rhs, p=2, dim=-1).mean()  # another stopping criterion
                stopping_criterion = eq_mean / eq_scale_mean  # another stopping criterion
                if stopping_criterion <= f_tol:  # another stopping criterion
                    break  # another stopping criterion

            z_star = z
            ctx.save_for_backward(J_accum)
        return z_star

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        J_accum, = ctx.saved_tensors
        grad_input = torch.bmm(grad_output.unsqueeze(1), J_accum).squeeze(1)
        return grad_input, None, None, None, None, None, None, None, None


class PrimalNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, output_dim,
                 truncate_idx, free_idx, A, WzProj, WbProj,
                 max_iter, f_tol, report_projection=True):
        super(PrimalNN, self).__init__()

        self.output_dim = output_dim
        self.truncate_idx = truncate_idx
        self.free_num = free_idx[1] + 1  # dataset has been modified to have free variables at the beginning.

        # optimality layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(hidden_num - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        # feasibility layers
        self.A = A.requires_grad_(False)
        self.WzProj = WzProj.requires_grad_(False)
        self.WbProj = WbProj.requires_grad_(False)

        self.max_iter = max_iter
        self.f_tol = f_tol

        self.report_projection = report_projection

    def forward(self, b):
        b_0 = b  # store the original inputs (not truncated)
        b = b[:, self.truncate_idx[0]:self.truncate_idx[1]]
        for layer in self.layers[:-1]:
            b = F.relu(layer(b))
        out = self.layers[-1](b)

        if self.training:
            projections = ProjectionIter.apply
            z_star = projections(out, self.A, self.WzProj, self.WbProj, b_0,
                                 self.max_iter, self.f_tol, self.output_dim, self.free_num)
            return z_star, out

        else:
            with torch.no_grad():
                z = out
                curr_iter = 1

                Bias_Proj = torch.matmul(b_0, self.WbProj.t())
                while curr_iter <= self.max_iter:
                    z = Bias_Proj + torch.matmul(z, self.WzProj.t())

                    u, v = torch.split(z, [self.free_num, self.output_dim - self.free_num], dim=-1)
                    v = F.relu(v)
                    z = torch.cat((u, v), dim=-1)

                    curr_iter = curr_iter + 1
                    eq_residual = z @ self.A.t() - b_0
                    # worst_of_worst = torch.norm(eq_residual, float('inf')).item()
                    # if worst_of_worst <= self.f_tol:
                    #     break
                    eq_rhs = b_0  # another stopping criterion
                    eq_mean = torch.norm(eq_residual, p=2, dim=-1).mean()  # another stopping criterion
                    eq_scale_mean = 1 + torch.norm(eq_rhs, p=2, dim=-1).mean()  # another stopping criterion
                    stopping_criterion = eq_mean / eq_scale_mean  # another stopping criterion
                    if stopping_criterion <= self.f_tol:  # another stopping criterion
                        break  # another stopping criterion

                z_star = z

                if self.report_projection:
                    return z_star, out, curr_iter
                else:
                    return z_star, out


class DualNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, output_dim,
                 truncate_idx, free_idx, A, b, WzProj, WbProj,
                 max_iter, f_tol, report_projection=True):
        super(DualNN, self).__init__()

        self.output_dim = output_dim
        self.truncate_idx = truncate_idx
        self.free_num = free_idx[1] + 1  # dataset has been modified to have free variables at the beginning.

        # optimality layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(hidden_num - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        # feasibility layers
        self.A = A.requires_grad_(False)
        self.b = b(None).requires_grad_(False)
        self.WzProj = WzProj.requires_grad_(False)
        self.WbProj = WbProj.requires_grad_(False)

        self.max_iter = max_iter
        self.f_tol = f_tol

        self.report_projection = report_projection

    def forward(self, b_primal):
        b_primal = b_primal[:, self.truncate_idx[0]:self.truncate_idx[1]]
        for layer in self.layers[:-1]:
            b_primal = F.relu(layer(b_primal))
        out = self.layers[-1](b_primal)

        if self.training:
            projections = ProjectionIter.apply
            z_star = projections(out, self.A, self.WzProj, self.WbProj, self.b,
                                 self.max_iter, self.f_tol, self.output_dim, self.free_num)
            return z_star, out

        else:
            with torch.no_grad():
                z = out
                curr_iter = 1

                Bias_Proj = torch.matmul(self.b, self.WbProj.t())
                while curr_iter <= self.max_iter:
                    z = Bias_Proj + torch.matmul(z, self.WzProj.t())

                    u, v = torch.split(z, [self.free_num, self.output_dim - self.free_num], dim=-1)
                    v = F.relu(v)
                    z = torch.cat((u, v), dim=-1)

                    curr_iter = curr_iter + 1
                    eq_residual = z @ self.A.t() - self.b
                    # worst_of_worst = torch.norm(eq_residual, float('inf')).item()
                    # if worst_of_worst <= self.f_tol:
                    #     break
                    eq_rhs = self.b  # another stopping criterion
                    eq_mean = torch.norm(eq_residual, p=2, dim=-1).mean()  # another stopping criterion
                    eq_scale_mean = 1 + torch.norm(eq_rhs, p=2, dim=-1).mean()  # another stopping criterion
                    stopping_criterion = eq_mean / eq_scale_mean  # another stopping criterion
                    if stopping_criterion <= self.f_tol:  # another stopping criterion
                        break  # another stopping criterion

                z_star = z

                if self.report_projection:
                    return z_star, out, curr_iter
                else:
                    return z_star, out


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
            return out, out
        else:
            return out, out, 0


class DualLearn2Proj(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, output_dim,
                 proj_hidden_dim, proj_hidden_num,
                 truncate_idx, free_idx, A, b, WzProj, WbProj,
                 max_iter, f_tol, report_projection=True):
        super(DualLearn2Proj, self).__init__()

        self.output_dim = output_dim
        self.truncate_idx = truncate_idx
        self.free_num = free_idx[1] + 1  # dataset has been modified to have free variables at the beginning.

        # optimality layers
        self.optimality_layers = nn.ModuleList()
        self.optimality_layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(hidden_num - 1):
            self.optimality_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.optimality_layers.append(nn.Linear(hidden_dim, output_dim))

        # (pseudo) - projection layers
        self.projection_layers = nn.ModuleList()
        self.projection_layers.append(nn.Linear(output_dim, proj_hidden_dim))
        self.projection_layers.append(nn.LayerNorm(proj_hidden_dim))  # add layer norm
        for _ in range(proj_hidden_num - 1):
            self.projection_layers.append(nn.Linear(proj_hidden_dim, proj_hidden_dim))
            self.projection_layers.append(nn.LayerNorm(proj_hidden_dim))  # add layer norm
        self.projection_layers.append(nn.Linear(proj_hidden_dim, output_dim))

        # feasibility layers
        self.A = A.requires_grad_(False)
        self.b = b(None).requires_grad_(False)
        self.WzProj = WzProj.requires_grad_(False)
        self.WbProj = WbProj.requires_grad_(False)

        self.max_iter = max_iter
        self.f_tol = f_tol

        self.report_projection = report_projection

    def fix_projection_weights(self):
        for param in self.projection_layers.parameters():
            param.requires_grad = False

    def unfix_projection_weights(self):
        for param in self.projection_layers.parameters():
            param.requires_grad = True

    def forward(self, inputs, phase='opt+proj+feas'):
        if phase == 'opt+feas':
            # phase 1: create projection dataset
            # inputs are b_primal
            self.fix_projection_weights()
            inputs = inputs[:, self.truncate_idx[0]:self.truncate_idx[1]]
            for layer in self.optimality_layers[:-1]:
                inputs = F.relu(layer(inputs))
            z1 = self.optimality_layers[-1](inputs)

            with torch.no_grad():
                z = z1
                curr_iter = 1

                Bias_Proj = torch.matmul(self.b, self.WbProj.t())
                while curr_iter <= self.max_iter:
                    z = Bias_Proj + torch.matmul(z, self.WzProj.t())

                    u, v = torch.split(z, [self.free_num, self.output_dim - self.free_num], dim=-1)
                    v = F.relu(v)
                    z = torch.cat((u, v), dim=-1)

                    curr_iter = curr_iter + 1
                    eq_residual = z @ self.A.t() - self.b
                    # worst_of_worst = torch.norm(eq_residual, float('inf')).item()
                    # if worst_of_worst <= self.f_tol:
                    #     break
                    eq_rhs = self.b  # another stopping criterion
                    eq_mean = torch.norm(eq_residual, p=2, dim=-1).mean()  # another stopping criterion
                    eq_scale_mean = 1 + torch.norm(eq_rhs, p=2, dim=-1).mean()  # another stopping criterion
                    stopping_criterion = eq_mean / eq_scale_mean  # another stopping criterion
                    if stopping_criterion <= self.f_tol:  # another stopping criterion
                        break  # another stopping criterion

                z_star = z
                if self.report_projection:
                    return z_star, z1, curr_iter
                else:
                    return z_star, z1

        elif phase == 'proj':
            # phase 2: learn projection
            # inputs are z1
            self.unfix_projection_weights()

            for i in range(0, len(self.projection_layers) - 1, 2):
                inputs = F.relu(self.projection_layers[i](inputs))
                inputs = self.projection_layers[i+1](inputs)  # Apply LayerNorm
            pseudo_z_star = self.projection_layers[-1](inputs)

            u, v = torch.split(pseudo_z_star, [self.free_num, self.output_dim - self.free_num], dim=-1)
            v = F.relu(v)
            pseudo_z_star = torch.cat((u, v), dim=-1)
            return pseudo_z_star

        elif phase == 'opt+proj+feas':
            # phase 3: learn optimality
            # inputs are b_primal
            self.fix_projection_weights()
            inputs = inputs[:, self.truncate_idx[0]:self.truncate_idx[1]]
            for layer in self.optimality_layers[:-1]:
                inputs = F.relu(layer(inputs))
            z1 = self.optimality_layers[-1](inputs)

            for i in range(0, len(self.projection_layers) - 1, 2):
                z1 = F.relu(self.projection_layers[i](z1))
                z1 = self.projection_layers[i+1](z1)  # Apply LayerNorm
            pseudo_z_star = self.projection_layers[-1](z1)

            u, v = torch.split(pseudo_z_star, [self.free_num, self.output_dim - self.free_num], dim=-1)
            v = F.relu(v)
            pseudo_z_star = torch.cat((u, v), dim=-1)

            if self.training:
                projections = ProjectionIter.apply
                z_star = projections(pseudo_z_star, self.A, self.WzProj, self.WbProj, self.b,
                                     self.max_iter, self.f_tol, self.output_dim, self.free_num)
                return z_star, pseudo_z_star
            else:
                with torch.no_grad():
                    z = pseudo_z_star
                    curr_iter = 1

                    Bias_Proj = torch.matmul(self.b, self.WbProj.t())
                    while curr_iter <= self.max_iter:
                        z = Bias_Proj + torch.matmul(z, self.WzProj.t())

                        u, v = torch.split(z, [self.free_num, self.output_dim - self.free_num], dim=-1)
                        v = F.relu(v)
                        z = torch.cat((u, v), dim=-1)

                        curr_iter = curr_iter + 1
                        eq_residual = z @ self.A.t() - self.b
                        # worst_of_worst = torch.norm(eq_residual, float('inf')).item()
                        # if worst_of_worst <= self.f_tol:
                        #     break
                        eq_rhs = self.b  # another stopping criterion
                        eq_mean = torch.norm(eq_residual, p=2, dim=-1).mean()  # another stopping criterion
                        eq_scale_mean = 1 + torch.norm(eq_rhs, p=2, dim=-1).mean()  # another stopping criterion
                        stopping_criterion = eq_mean / eq_scale_mean  # another stopping criterion
                        if stopping_criterion <= self.f_tol:  # another stopping criterion
                            break  # another stopping criterion

                    z_star = z
                    if self.report_projection:
                        return z_star, pseudo_z_star, curr_iter
                    else:
                        return z_star, pseudo_z_star

        else:
            raise ValueError('Invalid phase')




