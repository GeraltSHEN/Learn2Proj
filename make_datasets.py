from utils import *
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from train import load_weights
# import osqp
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=64)


# def get_and_save_proj_data(model, custom_solver, args):
#     pass
#     for test_val_train in ['test', 'val', 'train']:
#         proj_targets = []
#         b_primal = torch.load('./data/' + args.dataset + f'/{test_val_train}/input_{test_val_train}.pt')
#         _, z1, __ = model(b_primal)
#         z1_data = TensorDataset(z1)
#         z1_dataloader = DataLoader(z1_data, batch_size=args.batch_size, shuffle=False)
#
#         total_time = 0
#         total_iters = 0
#         num_points = 0
#         for i, inputs in enumerate(z1_dataloader):
#             z1 = inputs[0]
#             start = time.time()
#             z_star_custom, curr_iter = custom_solver(z1)
#             proj_targets.append(z_star_custom)
#             end = time.time()
#
#             solver_time = end - start
#             total_time += solver_time
#             total_iters += curr_iter
#             num_points += 1
#
#         mean_time = total_time / num_points
#         mean_iters = total_iters / num_points
#         print(f'{test_val_train} data: '
#               f'Mean time: {mean_time}, Mean projections: {mean_iters}')
#         proj_targets_data = torch.cat(proj_targets, dim=0)
#         torch.save(z1.detach().cpu(), './data/' + args.dataset + f'/{test_val_train}/proj_input_{test_val_train}.pt')
#         torch.save(proj_targets_data.detach().cpu(), './data/' + args.dataset + f'/{test_val_train}/proj_target_{test_val_train}.pt')




# def make_proj_dataset(args):
#     pass
#     problem = load_problem(args)
#     assert 'dual' in problem.name
#     model = load_model(args, problem)
#     load_weights(model, args.model_id)
#     Wz_proj, Wb_proj = load_W_proj(args, problem)
#     print(f'Creating inputs')
#     custom_solver = ProjSolver(model.output_dim, args.dual_fx_idx,
#                                problem.A, problem.b, Wz_proj, Wb_proj,
#                                args.max_iter, args.f_tol)
#     model.eval()
#     custom_solver.eval()
#     get_and_save_proj_data(model, custom_solver, args)


# class ProjSolver(nn.Module):
#     def __init__(self, output_dim, free_idx, A, b, WzProj, WbProj, max_iter, f_tol):
#         super(ProjSolver, self).__init__()
#         self.output_dim = output_dim
#         self.free_num = free_idx[1] + 1  # dataset has been modified to have free variables at the beginning.
#         self.A = A
#         self.b = b
#         self.WzProj = WzProj
#         self.WbProj = WbProj
#         self.max_iter = max_iter
#         self.f_tol = f_tol
#
#     def forward(self, z):
#         with torch.no_grad():
#             curr_iter = 1
#             Bias_Proj = torch.matmul(self.b, self.WbProj.t())
#             while curr_iter <= self.max_iter:
#                 z = Bias_Proj + torch.matmul(z, self.WzProj.t())
#
#                 u, v = torch.split(z, [self.free_num, self.output_dim - self.free_num], dim=-1)
#                 v = F.relu(v)
#                 z = torch.cat((u, v), dim=-1)
#
#                 curr_iter = curr_iter + 1
#                 eq_residual = z @ self.A.t() - self.b
#                 worst_of_worst = torch.norm(eq_residual, float('inf')).item()
#
#                 equality = torch.norm(self.b -
#                                       torch.matmul(z, self.A.t()),
#                                       float('inf')).item()
#                 if worst_of_worst <= self.f_tol:
#                     break
#             z_star = z
#             return z_star, curr_iter


# def proj_sanity_check(args):
#     pass
#     problem = load_problem(args)
#     assert 'dual' in problem.name
#     model = load_model(args, problem)
#     load_weights(model, args.model_id)
#     Wz_proj, Wb_proj = load_W_proj(args, problem)
#     b_primal = torch.load('./data/' + args.dataset + '/train/input_train.pt')
#     model.eval()
#     _, z1, __ = model(b_primal)
#     z1_data = TensorDataset(z1)
#     z1_dataloader = DataLoader(z1_data, batch_size=1, shuffle=False)
#
#     custom_solver = ProjSolver(model.output_dim, args.dual_fx_idx,
#                                problem.A, problem.b, Wz_proj, Wb_proj,
#                                args.max_iter, args.f_tol)
#     # # osqp solver
#     # P = sparse.eye(problem.var_num, format='csc')
#     # q = np.zeros(problem.var_num)
#     # A = sparse.csc_matrix(np.vstack([problem.A.numpy(), problem.s_W.numpy()]))
#     # l = np.hstack([problem.b.numpy(), np.zeros(problem.var_num - problem.free_num)])
#     # u = np.hstack([problem.b.numpy(), np.inf * np.ones(problem.var_num - problem.free_num)])
#     # prob = osqp.OSQP()
#     # prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)
#     # prob.solve()
#
#     total_time = 0
#     total_iters = 0
#     num_points = 0
#     for i, inputs in enumerate(z1_dataloader):
#         z1 = inputs[0]
#         start = time.time()
#         z_star_custom, curr_iter = custom_solver(z1)
#         end = time.time()
#
#         solver_time = end - start
#         total_time += solver_time
#         total_iters += curr_iter
#         num_points += 1
#
#         if i % 1000 == 0:
#             print(f'Custom solver time for point {i}: {solver_time}')
#             print(f'Custom solver iterations for point {i}: {curr_iter}')
#         # start = time.time()
#         # prob.update(q= - z1.squeeze().detach().numpy())
#         z_star_custom = z_star_custom.squeeze().detach().numpy()
#         # z_star_osqp = prob.solve().x
#         assert z_star_custom.dtype == 'float64'
#         # assert z_star_osqp.dtype == 'float64'
#         # print(f'OSQP solver time: {time.time() - start}')
#         # projection_error = np.linalg.norm((z_star_custom - z_star_osqp) / z_star_osqp)
#         # print(f'train example {i}, projection error: {projection_error}')
#         eq_residual_custom = problem.eq_residual(torch.tensor(z_star_custom), None)
#         ineq_residual_custom = problem.ineq_residual(torch.tensor(z_star_custom))
#         eq_max_custom = torch.norm(eq_residual_custom, float('inf'))
#         ineq_max_custom = torch.norm(ineq_residual_custom, float('inf'))
#         if eq_max_custom > args.f_tol or ineq_max_custom > args.f_tol:
#             print('train bad point {: .0f}: eq max violation: {: .5f}; ineq max violation: {: .5f}'.format(
#                 i, eq_max_custom, ineq_max_custom))
#         # eq_residual_osqp = problem.eq_residual(torch.tensor(z_star_osqp), None)
#         # ineq_residual_osqp = problem.ineq_residual(torch.tensor(z_star_osqp))
#         # eq_max_osqp = torch.norm(eq_residual_osqp, float('inf'))
#         # ineq_max_osqp = torch.norm(ineq_residual_osqp, float('inf'))
#         # print(f'eq residual osqp: {eq_max_osqp}, ineq residual osqp: {ineq_max_osqp}')
#         #print(z_star_custom - z1.squeeze().detach().numpy() )
#         # print('z_star_custom @ A.t() \n', z_star_custom @ A.T)
#         # print('l \n', l)
#         # print('u \n', u)
#         # print('problem.b \n', problem.b)
#         # print('problem.A \n', problem.A)
#         # print('A \n', A)
#     mean_time = total_time / num_points
#     mean_iters = total_iters / num_points
#     print(f'Mean time: {mean_time}, Mean projections: {mean_iters}')


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
# A = sparse.csc_matrix(np.vstack([problem.A.numpy(), problem.s_W.numpy()]))
# A_not_sparse = np.vstack([problem.A.numpy(), problem.s_W.numpy()])
# l = np.hstack([problem.b.numpy(), np.zeros(problem.var_num - problem.free_num)])
# u = np.hstack([problem.b.numpy(), np.inf * np.ones(problem.var_num - problem.free_num)])
# prob = osqp.OSQP()
# prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=True)
# z_star_osqp = prob.solve().x







