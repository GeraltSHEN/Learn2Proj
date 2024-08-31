from utils import load_problem_new, load_data_new, load_W_proj, load_N, process_for_training
import torch
import models

import csv
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import random
import time
import os
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)


def load_solvers(args, problem):
    Wz_proj, Wb_proj = load_W_proj(args, problem)
    N = load_N(args, problem)
    pocs_solver = models.POCS(problem.free_idx, problem.A, Wz_proj, args.max_iter, args.eq_tol, args.ineq_tol)

    if args.periodic:
        eapm_solver = models.PeriodicEAPM(problem.free_idx, problem.A, Wz_proj, args.max_iter, args.eq_tol, args.ineq_tol, args.rho)
    else:
        eapm_solver = models.EAPM(problem.free_idx, problem.A, Wz_proj, args.max_iter, args.eq_tol, args.ineq_tol, args.rho)

    nullspace_solver = models.NullSpace(problem.free_idx, problem.A, N, Wz_proj, args.max_iter, args.eq_tol, args.ineq_tol, args.rho)
    return pocs_solver, eapm_solver, nullspace_solver, Wb_proj.t().requires_grad_(False).to(device)


def data_sanity_check(args):
    # warning: this function needs true labels of x_primal and y_dual
    pass
    # data_types = ['train', 'val', 'test']
    # for data_type in data_types:
    #     data, problem = load_data(args, data_type)
    #     assert problem.A.dtype == torch.float64
    #     for i, (inputs, targets) in enumerate(data):
    #         inputs, targets = process_for_training(inputs, targets, args)
    #         assert inputs.dtype == torch.float64
    #         assert targets.dtype == torch.float64
    #         eq_residual = problem.eq_residual(targets, inputs)
    #         ineq_residual = problem.ineq_residual(targets)
    #         assert eq_residual.dtype == torch.float64
    #         assert ineq_residual.dtype == torch.float64
    #         eq_stopping_criterion = torch.mean(torch.abs(eq_residual), dim=0)
    #         ineq_stopping_criterion = torch.mean(torch.abs(ineq_residual), dim=0)
    #         if (eq_stopping_criterion > args.eq_tol).all() or (ineq_stopping_criterion > args.ineq_tol).all():
    #             print('{} bad point {: .0f}: eq max violation: {: .5f}; ineq max violation: {: .5f}'.format(
    #                 data_type, i, eq_stopping_criterion.max(), ineq_stopping_criterion.max()))


def projection_on_data(data, solver, Wb_proj, args):
    solver.eval()
    avg_proj_num = 0
    unconverged = 0
    unconverged_idx = []
    ineq_violation = 0
    eq_violation = 0
    total_time = 0

    for i, (inputs, targets) in enumerate(data):
        inputs, targets = process_for_training(inputs, targets, args)
        with torch.no_grad():
            zero_z = torch.zeros_like(targets)
            b_primal = inputs
            Bias_Proj = b_primal @ Wb_proj  # has been transposed
        # let's count the projection time
        start = time.time()
        z_star, proj_num = solver(zero_z, Bias_Proj, b_primal)
        total_time = total_time + time.time() - start
        ineq_violation += solver.stopping_criterion(z_star, b_primal)
        eq_violation += torch.mean(torch.abs(z_star @ solver.A.t() - b_primal), dim=0)

        if proj_num >= args.max_iter:
            unconverged += 1
            unconverged_idx.append(i)
            print(f'Projection not converged for example {i}')
        avg_proj_num += proj_num
    avg_proj_num /= len(data)
    unconverged_rate = unconverged / len(data)
    total_time /= len(data)
    ineq_violation /= len(data)
    eq_violation /= len(data)

    print(f'Average projection number for train: {avg_proj_num:.2f}, '
          f'unconverged rate: {unconverged_rate:.5f}, '
          f'projection time per instance: {total_time:.2f} seconds, '
          f'inequality violation: {ineq_violation}, '
          f'equality violation: {eq_violation}')
    return avg_proj_num, unconverged_rate, unconverged_idx, total_time, ineq_violation, eq_violation


def baseline_pocs(args):
    dictionary = {'dataset': args.dataset,
                  'precond': args.precondition,
                  'projection': 'pocs',
                  'eq_tol': args.eq_tol,
                  'ineq_tol': args.ineq_tol,
                  'max_iter': args.max_iter}
    problem = load_problem_new(args)
    data = load_data_new(args)
    data_types = ['train', 'val', 'test']
    for data_type in data_types:
        pocs_solver, _, _, Wb_proj = load_solvers(args, problem)
        avg_proj_num, unconverged_rate, unconverged_idx, proj_time, ineq_violation, eq_violation = projection_on_data(data[data_type], pocs_solver, Wb_proj, args)
        dictionary[f'avg proj num {data_type}'] = avg_proj_num
        dictionary[f'unconverged rate {data_type}'] = unconverged_rate
        dictionary[f'avg proj time {data_type}'] = proj_time
    with open(f'./data/sanity_check/{args.model_id}_{args.dataset}_{args.precondition}_pocs_baseline.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['avg proj num train', 'unconverged rate train', 'avg proj time train',
                         'avg proj num val', 'unconverged rate val', 'avg proj time val',
                         'avg proj num test', 'unconverged rate test', 'avg proj time test',
                         'avg proj num total', 'unconverged rate total', 'avg proj time total'
                         ])
        writer.writerow([int(dictionary['avg proj num train']),
                         dictionary['unconverged rate train'],
                            dictionary['avg proj time train'],
                         int(dictionary['avg proj num val']),
                         dictionary['unconverged rate val'],
                            dictionary['avg proj time val'],
                         int(dictionary['avg proj num test']),
                         dictionary['unconverged rate test'],
                            dictionary['avg proj time test'],
                         int((dictionary['avg proj num train'] +
                              dictionary['avg proj num val'] +
                              dictionary['avg proj num test']) / 3),
                         (dictionary['unconverged rate train'] +
                          dictionary['unconverged rate val'] +
                          dictionary['unconverged rate test']) / 3,
                            (dictionary['avg proj time train'] +
                             dictionary['avg proj time val'] +
                             dictionary['avg proj time test']) / 3])
        writer.writerow([])
        writer.writerow([])
        writer.writerow(['dataset', args.dataset])
        writer.writerow(['precondition', args.precondition])
        writer.writerow(['projection', 'pocs'])
        writer.writerow(['eq_tol', args.eq_tol])
        writer.writerow(['ineq_tol', args.ineq_tol])
        writer.writerow(['max_iter', args.max_iter])
        writer.writerow(['device', device])


def update_rho_search_dict(data, data_type, solver, Wb_proj, args, dictionary):
    rho_sub_dict = {}
    print(f'----- checking rho = {args.rho}, {data_type}')
    avg_proj_num, unconverged_rate, unconverged_idx, proj_time, ineq_violation, eq_violation = projection_on_data(data, solver, Wb_proj, args)
    rho_sub_dict[f'avg proj num {data_type}'] = avg_proj_num
    rho_sub_dict[f'unconverged rate {data_type}'] = unconverged_rate
    rho_sub_dict[f'avg proj time {data_type}'] = proj_time
    dictionary['rho'][f'{args.rho}'].update(rho_sub_dict)


def rho_search(args):
    dictionary = {'dataset': args.dataset,
                  'precond': args.precondition,
                  'projection': 'eapm',
                  'eq_tol': args.eq_tol,
                  'ineq_tol': args.ineq_tol,
                  'max_iter': args.max_iter,
                  'periodic': args.periodic,
                  'rho': {}}

    problem = load_problem_new(args)
    data = load_data_new(args)
    data_types = ['train', 'val', 'test']

    for rho in [0.50, 1.00, 1.50, 1.75, 1.90]:
        args.rho = rho
        dictionary['rho'][f'{args.rho}'] = {}
        for data_type in data_types:
            _, eapm_solver, _, Wb_proj = load_solvers(args, problem)
            print(f'eapm_solver.rho = {eapm_solver.rho}')
            update_rho_search_dict(data[data_type], data_type, eapm_solver, Wb_proj, args, dictionary)

    # construct a csv where each column is a different rho
    with open(f'./data/sanity_check/{args.model_id}_{args.dataset}_{args.precondition}{args.periodic}_rho_search.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['rho',
                         'avg proj num train', 'unconverged rate train', 'avg proj time train',
                         'avg proj num val', 'unconverged rate val', 'avg proj time val',
                         'avg proj num test', 'unconverged rate test', 'avg proj time test',
                         'avg proj num total', 'unconverged rate total', 'avg proj time total'
                         ])
        for rho in dictionary['rho'].keys():
            writer.writerow([rho,
                             int(dictionary['rho'][rho]['avg proj num train']),
                             dictionary['rho'][rho]['unconverged rate train'],
                             dictionary['rho'][rho]['avg proj time train'],
                             int(dictionary['rho'][rho]['avg proj num val']),
                             dictionary['rho'][rho]['unconverged rate val'],
                             dictionary['rho'][rho]['avg proj time val'],
                             int(dictionary['rho'][rho]['avg proj num test']),
                             dictionary['rho'][rho]['unconverged rate test'],
                             dictionary['rho'][rho]['avg proj time test'],
                             int((dictionary['rho'][rho]['avg proj num train'] +
                                  dictionary['rho'][rho]['avg proj num val'] +
                                  dictionary['rho'][rho]['avg proj num test']) / 3),
                             (dictionary['rho'][rho]['unconverged rate train'] +
                              dictionary['rho'][rho]['unconverged rate val'] +
                              dictionary['rho'][rho]['unconverged rate test']) / 3,
                             (dictionary['rho'][rho]['avg proj time train'] +
                              dictionary['rho'][rho]['avg proj time val'] +
                              dictionary['rho'][rho]['avg proj time test']) / 3])
        # having two empty rows for better readability
        writer.writerow([])
        writer.writerow([])
        writer.writerow(['dataset', args.dataset])
        writer.writerow(['precondition', args.precondition])
        writer.writerow(['projection', 'eapm'])
        writer.writerow(['eq_tol', args.eq_tol])
        writer.writerow(['ineq_tol', args.ineq_tol])
        writer.writerow(['max_iter', args.max_iter])
        writer.writerow(['periodic', args.periodic])
        writer.writerow(['device', device])


def baseline_nullspace(args):
    pass
    dictionary = {'dataset': args.dataset,
                  'precond': args.precondition,
                  'projection': 'nullspace',
                  'eq_tol': args.eq_tol,
                  'ineq_tol': args.ineq_tol,
                  'max_iter': args.max_iter}
    problem = load_problem_new(args)
    data = load_data_new(args)
    data_types = ['train', 'val', 'test']
    for data_type in data_types:
        _, _, nullspace_solver, Wb_proj = load_solvers(args, problem)
        avg_proj_num, unconverged_rate, unconverged_idx, proj_time, ineq_violation, eq_violation = projection_on_data(data[data_type], nullspace_solver, Wb_proj, args)
        dictionary[f'avg proj num {data_type}'] = avg_proj_num
        dictionary[f'unconverged rate {data_type}'] = unconverged_rate
        dictionary[f'avg proj time {data_type}'] = proj_time
    with open(f'./data/sanity_check/{args.dataset}_{args.precondition}_nullspace_baseline.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['avg proj num train', 'unconverged rate train', 'avg proj time train',
                         'avg proj num val', 'unconverged rate val', 'avg proj time val',
                         'avg proj num test', 'unconverged rate test', 'avg proj time test',
                         'avg proj num total', 'unconverged rate total', 'avg proj time total'
                         ])
        writer.writerow([int(dictionary['avg proj num train']),
                         dictionary['unconverged rate train'],
                            dictionary['avg proj time train'],
                         int(dictionary['avg proj num val']),
                         dictionary['unconverged rate val'],
                            dictionary['avg proj time val'],
                         int(dictionary['avg proj num test']),
                         dictionary['unconverged rate test'],
                            dictionary['avg proj time test'],
                         int((dictionary['avg proj num train'] +
                              dictionary['avg proj num val'] +
                              dictionary['avg proj num test']) / 3),
                         (dictionary['unconverged rate train'] +
                          dictionary['unconverged rate val'] +
                          dictionary['unconverged rate test']) / 3,
                            (dictionary['avg proj time train'] +
                             dictionary['avg proj time val'] +
                             dictionary['avg proj time test']) / 3])
        writer.writerow([])
        writer.writerow([])
        writer.writerow(['dataset', args.dataset])
        writer.writerow(['precondition', args.precondition])
        writer.writerow(['projection', 'pocs'])
        writer.writerow(['eq_tol', args.eq_tol])
        writer.writerow(['ineq_tol', args.ineq_tol])
        writer.writerow(['max_iter', args.max_iter])
        writer.writerow(['device', device])

    # # todo: integrate nullspace into eapm
    # dictionary = {'dataset': args.dataset,
    #               'precond': args.precondition,
    #               'projection': 'nullspace',
    #               'eq_tol': args.eq_tol,
    #               'ineq_tol': args.ineq_tol,
    #               'max_iter': args.max_iter,
    #               'periodic': args.periodic,
    #               'rho': {}}
    #
    # problem = load_problem_new(args)
    # data = load_data_new(args)
    # data_types = ['train', 'val', 'test']
    #
    # for rho in [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.90]:
    #     args.rho = rho
    #     dictionary['rho'][f'{args.rho}'] = {}
    #     for data_type in data_types:
    #         _, _, nullspace_solver, Wb_proj = load_solvers(args, problem)
    #         print(f'eapm_solver.rho = {nullspace_solver.rho}')
    #         update_rho_search_dict(data[data_type], data_type, nullspace_solver, Wb_proj, args, dictionary)
    #
    # # construct a csv where each column is a different rho
    # with open(f'./data/sanity_check/{args.dataset}_{args.precondition}{args.periodic}_nullspace.csv', 'w',
    #           newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['rho',
    #                      'avg proj num train', 'unconverged rate train', 'avg proj time train',
    #                      'avg proj num val', 'unconverged rate val', 'avg proj time val',
    #                      'avg proj num test', 'unconverged rate test', 'avg proj time test',
    #                      'avg proj num total', 'unconverged rate total', 'avg proj time total'
    #                      ])
    #     for rho in dictionary['rho'].keys():
    #         writer.writerow([rho,
    #                          int(dictionary['rho'][rho]['avg proj num train']),
    #                          dictionary['rho'][rho]['unconverged rate train'],
    #                          dictionary['rho'][rho]['avg proj time train'],
    #                          int(dictionary['rho'][rho]['avg proj num val']),
    #                          dictionary['rho'][rho]['unconverged rate val'],
    #                          dictionary['rho'][rho]['avg proj time val'],
    #                          int(dictionary['rho'][rho]['avg proj num test']),
    #                          dictionary['rho'][rho]['unconverged rate test'],
    #                          dictionary['rho'][rho]['avg proj time test'],
    #                          int((dictionary['rho'][rho]['avg proj num train'] +
    #                               dictionary['rho'][rho]['avg proj num val'] +
    #                               dictionary['rho'][rho]['avg proj num test']) / 3),
    #                          (dictionary['rho'][rho]['unconverged rate train'] +
    #                           dictionary['rho'][rho]['unconverged rate val'] +
    #                           dictionary['rho'][rho]['unconverged rate test']) / 3,
    #                          (dictionary['rho'][rho]['avg proj time train'] +
    #                           dictionary['rho'][rho]['avg proj time val'] +
    #                           dictionary['rho'][rho]['avg proj time test']) / 3])
    #     # having two empty rows for better readability
    #     writer.writerow([])
    #     writer.writerow([])
    #     writer.writerow(['dataset', args.dataset])
    #     writer.writerow(['precondition', args.precondition])
    #     writer.writerow(['projection', 'eapm'])
    #     writer.writerow(['eq_tol', args.eq_tol])
    #     writer.writerow(['ineq_tol', args.ineq_tol])
    #     writer.writerow(['max_iter', args.max_iter])
    #     writer.writerow(['periodic', args.periodic])
    #     writer.writerow(['device', device])