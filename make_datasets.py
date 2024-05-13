
from utils import *
import numpy as np
from scipy import sparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from train import load_weights
import osqp
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=64)


def make_proj_dataset(args):
    problem = load_problem(args)
    assert 'dual' in problem.name
    model = load_model(args, problem)
    load_weights(model, args.model_id)
    print(f'Creating inputs')
    if args.data_generator:
        print('self-generated data training code has not been implemented yet')
    else:
        # models output will be collected as inputs
        print('using the outputs from the training data')
        b_primal = torch.load('./data/' + args.dataset + '/train/input_train.pt')
        _, z1, __ = model(b_primal)

    if not os.path.exists('./data/' + args.dataset + '/train/target_train.pt'):
        print('Creating targets ...')
    else:
        print('Targets already exist')


class ProjSolver(nn.Module):
    def __init__(self, output_dim, free_idx, A, b, WzProj, WbProj, max_iter, f_tol):
        super(ProjSolver, self).__init__()
        self.output_dim = output_dim
        self.free_num = free_idx[1] + 1  # dataset has been modified to have free variables at the beginning.
        self.A = A
        self.b = b
        self.WzProj = WzProj
        self.WbProj = WbProj
        self.max_iter = max_iter
        self.f_tol = f_tol

    def forward(self, z):
        curr_iter = 1
        Bias_Proj = torch.matmul(self.b, self.WbProj.t())
        while curr_iter <= self.max_iter:
            z = Bias_Proj + torch.matmul(z, self.WzProj.t())

            u, v = torch.split(z, [self.free_num, self.output_dim - self.free_num], dim=-1)
            v = F.relu(v)
            z = torch.cat((u, v), dim=-1)

            curr_iter = curr_iter + 1
            eq_residual = z @ self.A.t() - self.b
            worst_of_worst = torch.norm(eq_residual, float('inf')).item()

            equality = torch.norm(self.b -
                                  torch.matmul(z, self.A.t()),
                                  float('inf')).item()
            if worst_of_worst <= self.f_tol:
                break
        z_star = z
        return z_star, curr_iter


def sanity_check(args):
    problem = load_problem(args)
    assert 'dual' in problem.name
    model = load_model(args, problem)
    load_weights(model, args.model_id)
    Wz_proj, Wb_proj = load_W_proj(args, problem)
    b_primal = torch.load('./data/' + args.dataset + '/train/input_train.pt')
    model.eval()
    _, z1, __ = model(b_primal)
    z1_data = TensorDataset(z1)
    z1_dataloader = DataLoader(z1_data, batch_size=1, shuffle=False)

    custom_solver = ProjSolver(model.output_dim, args.dual_fx_idx,
                               problem.A, problem.b, Wz_proj, Wb_proj,
                               args.max_iter, args.f_tol)
    # osqp solver
    P = sparse.eye(problem.var_num, format='csc')
    q = np.zeros(problem.var_num)
    A = sparse.bmat(np.vstack([problem.A.numpy(), problem.s_W.numpy()]), format='csc')
    l = np.hstack([problem.b.numpy(), np.zeros(problem.var_num - problem.free_num)])
    u = np.hstack([problem.b.numpy(), np.inf * np.ones(problem.var_num - problem.free_num)])
    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=A, l=l, u=u)
    prob.solve()

    for i, inputs in enumerate(z1_dataloader):
        z1 = inputs[0]
        start = time.time()
        z_star_custom, curr_iter = custom_solver(z1)
        print(f'Custom solver time: {time.time() - start}')
        print(f'Custom solver iterations: {curr_iter}')
        start = time.time()
        prob.update(q=-2 * z1.numpy())
        z_star_osqp = prob.solve().x
        print(f'OSQP solver time: {time.time() - start}')
        break

# class args:
#     def __init__(self):
#         pass
# args = args()
# defaults = {'problem': 'dual_lp',
#             'dataset': 'DCOPF',
#             'dual_fx_idx': (0,39),
#             'truncate_idx': (2,40),
#             'max_iter': 1000,
#             'f_tol': 1e-4}
# for key in defaults.keys():
#     exec('args.' + key + ' = defaults[key]')
# problem = load_problem(args)
# Wz_proj, Wb_proj = load_W_proj(args, problem)
# b_primal = torch.load('./data/' + args.dataset + '/train/input_train.pt')
# custom_solver = ProjSolver(problem.var_num, args.dual_fx_idx,
#                                problem.A, problem.b, Wz_proj, Wb_proj,
#                                args.max_iter, args.f_tol)
# # osqp solver
# P = sparse.eye(problem.var_num, format='csc')
# q = np.zeros(problem.var_num)
# A = sparse.bmat(np.vstack([problem.A.numpy(), problem.s_W.numpy()]), format='csc')
# l = np.hstack([problem.b.numpy(), np.zeros(problem.var_num - problem.free_num)])
# u = np.hstack([problem.b.numpy(), np.inf * np.ones(problem.var_num - problem.free_num)])
# prob = osqp.OSQP()
# prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=True)
# res = prob.solve()
# print(f'first z1: {res.x}')







