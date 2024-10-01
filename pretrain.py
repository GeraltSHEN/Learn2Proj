from utils import load_problem_new, load_data_new, load_W_proj, load_LDR, load_N, process_for_training
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


def load_solvers(args, problem):
    Wz_proj, Wb_proj = load_W_proj(args, problem)
    Q_LDR, z0_LDR = load_LDR(args, problem)
    pocs_solver = models.POCS(problem.free_idx, problem.A, Wz_proj, args.max_iter, args.eq_tol, args.ineq_tol)
    ldr_solver = models.LDRPM(problem.free_idx, problem.A, Wz_proj, Q_LDR, z0_LDR, args.eq_tol, args.ineq_tol)

    if args.periodic:
        eapm_solver = models.PeriodicEAPM(problem.free_idx, problem.A, Wz_proj, args.max_iter, args.eq_tol, args.ineq_tol, args.rho)
    else:
        eapm_solver = models.EAPM(problem.free_idx, problem.A, Wz_proj, args.max_iter, args.eq_tol, args.ineq_tol, args.rho)

    return pocs_solver, eapm_solver, ldr_solver, Wb_proj.t().requires_grad_(False).to(device)


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
    measures = {'avg_proj_num': 0,
                'unconverged': 0,
                'unconverged_idx': [],
                'violation': 0,
                'ineq_violation': 0,
                'eq_violation': 0,
                'total_time': 0}

    for i, (inputs, targets) in enumerate(data):
        inputs, targets = process_for_training(inputs, targets, args)
        with torch.no_grad():
            rand_z = torch.randn_like(targets)
            b_primal = inputs
            Bias_Proj = b_primal @ Wb_proj  # has been transposed
        # let's count the projection time
        start = time.time()
        z_star, proj_num, alpha = solver(rand_z, Bias_Proj, b_primal)
        measures['total_time'] = measures['total_time'] + time.time() - start
        measures['avg_proj_num'] += proj_num
        measures['violation'] += solver.stopping_criterion(z_star, b_primal).item()
        measures['ineq_violation'] += torch.mean(torch.abs(torch.relu(-z_star[:, solver.free_num:])), dim=0)
        measures['eq_violation'] += torch.mean(torch.abs(z_star @ solver.A.t() - b_primal), dim=0)

        if proj_num >= args.max_iter:
            measures['unconverged'] += 1
            measures['unconverged_idx'].append(i)
            print(f'Projection not converged for example {i}')

    measures['avg_proj_num'] /= len(data)
    measures['unconverged_rate'] = measures['unconverged'] / len(data)
    measures['total_time'] /= len(data)
    measures['violation'] /= len(data)
    measures['ineq_violation'] /= len(data)
    measures['eq_violation'] /= len(data)
    measures['max_ineq_violation'] = measures['ineq_violation'].max().item()
    measures['num_ineq_violation'] = (measures['ineq_violation'] > args.ineq_tol).sum().item()
    measures['max_eq_violation'] = measures['eq_violation'].max().item()
    measures['num_eq_violation'] = (measures['eq_violation'] > args.eq_tol).sum().item()

    print(f'Average projection number for train: {measures["avg_proj_num"]:.2f}, '
          f'unconverged rate: {measures["unconverged_rate"]:.5f}, '
          f'projection time per instance: {measures["total_time"]:.2f} seconds, '
          f'violation: {measures["violation"]}, '
          f'max ineq violation: {measures["max_ineq_violation"]}, '
          f'num ineq violation: {measures["num_ineq_violation"]}, '
          f'max eq violation: {measures["max_eq_violation"]}, '
          f'num eq violation: {measures["num_eq_violation"]}')
    return measures


def baseline_pocs(args):
    dictionary = {'dataset': args.dataset,
                  'precond': args.precondition,
                  'projection': 'pocs',
                  'eq_tol': args.eq_tol,
                  'ineq_tol': args.ineq_tol,
                  'max_iter': args.max_iter}
    problem = load_problem_new(args)
    data = load_data_new(args, problem)
    data_types = ['train', 'val', 'test']
    for data_type in data_types:
        pocs_solver, _, _, Wb_proj = load_solvers(args, problem)
        avg_proj_num, unconverged_rate, unconverged_idx, proj_time, ineq_violation, eq_violation, violation = projection_on_data(data[data_type], pocs_solver, Wb_proj, args)
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
        writer.writerow(['float64', args.float64])


def run_proj_exp(args):
    measures_train_val_test = {}
    problem = load_problem_new(args)
    data = load_data_new(args, problem)
    data_types = ['train', 'val', 'test']
    for data_type in data_types:
        pocs_solver, eapm_solver, ldr_solver, Wb_proj = load_solvers(args, problem)
        if args.projection == 'POCS':
            solver = pocs_solver
        elif args.projection == 'EAPM':
            solver = eapm_solver
        elif args.projection == 'LDRPM':
            solver = ldr_solver
        else:
            raise ValueError('Unknown projection method')
        measures = projection_on_data(data[data_type], solver, Wb_proj, args)
        for key, value in measures.items():
            measures_train_val_test[f'{key} {data_type}'] = value

    measure_set = set(key.split()[0] for key in measures_train_val_test.keys())
    # remove unconverged idx
    measure_set.remove('unconverged_idx')
    for measure in measure_set:
        measures_train_val_test[f'{measure} total'] = (measures_train_val_test[f'{measure} train'] +
                                                       measures_train_val_test[f'{measure} val'] +
                                                       measures_train_val_test[f'{measure} test']) / 3
        if measure in ['avg_proj_num', 'num_ineq_violation', 'num_eq_violation']:
            measures_train_val_test[f'{measure} total'] = int(measures_train_val_test[f'{measure} total'])

    with open(f'./data/sanity_check/{args.model_id}_{args.dataset}_{args.precondition}_{args.projection}_ProjEXP.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        headers = list(measures_train_val_test.keys())
        headers.sort()
        writer.writerow(headers)
        writer.writerow([measures_train_val_test[key] for key in headers])

        writer.writerow([])
        writer.writerow([])
        writer.writerow(['dataset', args.dataset])
        writer.writerow(['precondition', args.precondition])
        writer.writerow(['projection', 'pocs'])
        writer.writerow(['eq_tol', args.eq_tol])
        writer.writerow(['ineq_tol', args.ineq_tol])
        writer.writerow(['max_iter', args.max_iter])
        writer.writerow(['device', device])
        writer.writerow(['float64', args.float64])



def update_rho_search_dict(data, data_type, solver, Wb_proj, args, dictionary):
    rho_sub_dict = {}
    print(f'----- checking rho = {args.rho}, {data_type}')
    avg_proj_num, unconverged_rate, unconverged_idx, proj_time, violation, max_ineq_violation, num_ineq_violation, max_eq_violation, num_eq_violation = projection_on_data(data, solver, Wb_proj, args)
    rho_sub_dict[f'avg proj num {data_type}'] = avg_proj_num
    rho_sub_dict[f'unconverged rate {data_type}'] = unconverged_rate
    rho_sub_dict[f'avg proj time {data_type}'] = proj_time
    rho_sub_dict[f'violation {data_type}'] = violation
    rho_sub_dict[f'max ineq violation {data_type}'] = max_ineq_violation
    rho_sub_dict[f'num ineq violation {data_type}'] = num_ineq_violation
    rho_sub_dict[f'max eq violation {data_type}'] = max_eq_violation
    rho_sub_dict[f'num eq violation {data_type}'] = num_eq_violation
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
    data = load_data_new(args, problem)
    data_types = ['train', 'val', 'test']

    # for rho in [0.50, 1.00, 1.50, 1.75, 1.90]:
    for rho in [1.90]:
        args.rho = rho
        dictionary['rho'][f'{args.rho}'] = {}
        for data_type in data_types:
            _, eapm_solver, _, Wb_proj = load_solvers(args, problem)
            print(f'eapm_solver.rho = {eapm_solver.rho}')
            update_rho_search_dict(data[data_type], data_type, eapm_solver, Wb_proj, args, dictionary)

    with open(f'./data/sanity_check/{args.model_id}_{args.dataset}_{args.precondition}{args.periodic}_rho_search.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        header = ['rho', 'avg proj num train', 'unconverged rate train', 'avg proj time train',
                  'avg proj num val', 'unconverged rate val', 'avg proj time val',
                  'avg proj num test', 'unconverged rate test', 'avg proj time test',
                  'avg proj num total', 'unconverged rate total', 'avg proj time total',
                  'max ineq violation train', 'num ineq violation train', 'max eq violation train', 'num eq violation train', 'violation train',
                  'max ineq violation val', 'num ineq violation val', 'max eq violation val', 'num eq violation val', 'violation val',
                  'max ineq violation test', 'num ineq violation test', 'max eq violation test', 'num eq violation test', 'violation test']
        writer.writerow(header)

        for rho, values in dictionary['rho'].items():
            avg_proj_num_total = int((values['avg proj num train'] + values['avg proj num val'] + values[
                'avg proj num test']) / 3)
            unconverged_rate_total = (values['unconverged rate train'] + values['unconverged rate val'] + values[
                'unconverged rate test']) / 3
            avg_proj_time_total = (values['avg proj time train'] + values['avg proj time val'] + values[
                'avg proj time test']) / 3

            row = [rho, int(values['avg proj num train']), values['unconverged rate train'],
                   values['avg proj time train'],
                   int(values['avg proj num val']), values['unconverged rate val'], values['avg proj time val'],
                   int(values['avg proj num test']), values['unconverged rate test'], values['avg proj time test'],
                   int(avg_proj_num_total), unconverged_rate_total, avg_proj_time_total,
                   values['max ineq violation train'], values['num ineq violation train'], values['max eq violation train'], values['num eq violation train'], values['violation train'],
                   values['max ineq violation val'], values['num ineq violation val'], values['max eq violation val'], values['num eq violation val'], values['violation val'],
                   values['max ineq violation test'], values['num ineq violation test'], values['max eq violation test'], values['num eq violation test'], values['violation test']]
            writer.writerow(row)

        writer.writerow([])
        writer.writerow([])

        metadata = [['dataset', args.dataset], ['precondition', args.precondition], ['projection', 'eapm'],
                    ['eq_tol', args.eq_tol], ['ineq_tol', args.ineq_tol], ['max_iter', args.max_iter],
                    ['periodic', args.periodic], ['device', device], ['float64', args.float64]]

        writer.writerows(metadata)


    # # construct a csv where each column is a different rho
    # with open(f'./data/sanity_check/{args.model_id}_{args.dataset}_{args.precondition}{args.periodic}_rho_search.csv', 'w', newline='') as file:
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
    #     writer.writerow(['float64', args.float64])

