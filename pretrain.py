from utils import load_problem, load_W_proj, process_for_training
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


def load_data(args, data_type):
    problem, D1, D2 = load_problem(args)
    assert problem.A.dtype == torch.float64

    if data_type == 'train':
        inputs = torch.load('./data/' + args.dataset + '/train/input_train.pt')
        inputs = inputs @ D1.t()
        targets = torch.load('./data/' + args.dataset + '/train/' + '' + 'target_train.pt')
    elif data_type == 'val':
        inputs = torch.load('./data/' + args.dataset + '/val/input_val.pt')
        inputs = inputs @ D1.t()
        targets = torch.load('./data/' + args.dataset + '/val/' + '' + 'target_val.pt')
    elif data_type == 'test':
        inputs = torch.load('./data/' + args.dataset + '/test/input_test.pt')
        inputs = inputs @ D1.t()
        targets = torch.load('./data/' + args.dataset + '/test/' + '' + 'target_test.pt')
    else:
        raise ValueError('data_type should be train, val, or test')
    dataset = TensorDataset(inputs, targets)
    data = DataLoader(dataset, batch_size=1, shuffle=False)
    return data, problem


def load_solvers(args, problem):
    Wz_proj, Wb_proj = load_W_proj(args, problem)
    pocs_solver = models.POCS(problem.free_idx, problem.A, Wz_proj, args.max_iter, args.eq_tol, args.ineq_tol)

    if args.periodic:
        eapm_solver = models.PeriodicEAPM(problem.free_idx, problem.A, Wz_proj, args.max_iter, args.eq_tol, args.ineq_tol, args.rho)
    else:
        eapm_solver = models.EAPM(problem.free_idx, problem.A, Wz_proj, args.max_iter, args.eq_tol, args.ineq_tol, args.rho)
    return pocs_solver, eapm_solver, Wb_proj.t().requires_grad_(False).to(device)


def projection_on_data(data, solver, Wb_proj, args):
    solver.eval()
    avg_proj_num = 0
    unconverged = 0
    unconverged_idx = []

    for i, (inputs, targets) in enumerate(data):
        inputs, targets = process_for_training(inputs, targets, args)
        with torch.no_grad():
            zero_z = torch.zeros_like(targets)
            b_primal = inputs
            Bias_Proj = b_primal @ Wb_proj  # has been transposed
        z_star, proj_num = solver(zero_z, Bias_Proj, b_primal)
        if proj_num >= args.max_iter:
            unconverged += 1
            unconverged_idx.append(i)
            print(f'Projection not converged for example {i}')
        avg_proj_num += proj_num
    avg_proj_num /= len(data)
    unconverged_rate = unconverged / len(data)
    print(f'Average projection number for train: {avg_proj_num:.2f}, '
          f'unconverged rate: {unconverged_rate:.5f}')
    return avg_proj_num, unconverged_rate, unconverged_idx


def baseline_pocs(args):
    dictionary = {'dataset': args.dataset,
                  'precond': args.precondition,
                  'projection': 'pocs',
                  'eq_tol': args.eq_tol,
                  'ineq_tol': args.ineq_tol,
                  'max_iter': args.max_iter}
    data_types = ['train', 'val', 'test']
    for data_type in data_types:
        data, problem = load_data(args, data_type)
        pocs_solver, _, Wb_proj = load_solvers(args, problem)
        avg_proj_num, unconverged_rate, unconverged_idx = projection_on_data(data, pocs_solver, Wb_proj, args)
        dictionary[f'avg proj num {data_type}'] = avg_proj_num
        dictionary[f'unconverged rate {data_type}'] = unconverged_rate
    with open(f'./data/sanity_check/{args.dataset}_{args.precondition}_pocs_baseline.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['avg proj num train', 'unconverged rate train',
                         'avg proj num val', 'unconverged rate val',
                         'avg proj num test', 'unconverged rate test',
                         'avg proj num total', 'unconverged rate total'])
        writer.writerow([int(dictionary['avg proj num train']),
                         dictionary['unconverged rate train'],
                         int(dictionary['avg proj num val']),
                         dictionary['unconverged rate val'],
                         int(dictionary['avg proj num test']),
                         dictionary['unconverged rate test'],
                         int((dictionary['avg proj num train'] +
                              dictionary['avg proj num val'] +
                              dictionary['avg proj num test']) / 3),
                         (dictionary['unconverged rate train'] +
                          dictionary['unconverged rate val'] +
                          dictionary['unconverged rate test']) / 3])
        writer.writerow([])
        writer.writerow([])
        writer.writerow(['dataset', args.dataset])
        writer.writerow(['precondition', args.precondition])
        writer.writerow(['projection', 'pocs'])
        writer.writerow(['eq_tol', args.eq_tol])
        writer.writerow(['ineq_tol', args.ineq_tol])
        writer.writerow(['max_iter', args.max_iter])


def update_rho_search_dict(data, data_type, eapm_solver, Wb_proj, args, dictionary):
    rho_sub_dict = {}
    print(f'----- checking rho = {args.rho}, {data_type}')
    avg_proj_num, unconverged_rate, unconverged_idx = projection_on_data(data, eapm_solver, Wb_proj, args)
    rho_sub_dict[f'avg proj num {data_type}'] = avg_proj_num
    rho_sub_dict[f'unconverged rate {data_type}'] = unconverged_rate
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
    data_types = ['train', 'val', 'test']
    for rho in [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75]:
        args.rho = rho
        dictionary['rho'][f'{args.rho}'] = {}
        for data_type in data_types:
            data, problem = load_data(args, data_type)
            _, eapm_solver, Wb_proj = load_solvers(args, problem)
            print(f'eapm_solver.rho = {eapm_solver.rho}')
            update_rho_search_dict(data, data_type, eapm_solver, Wb_proj, args, dictionary)
    # construct a csv where each column is a different rho
    with open(f'./data/sanity_check/{args.dataset}_{args.precondition}_rho_search.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['rho', 'avg proj num train', 'unconverged rate train',
                         'avg proj num val', 'unconverged rate val',
                         'avg proj num test', 'unconverged rate test',
                         'avg proj num total', 'unconverged rate total'])
        for rho in dictionary['rho'].keys():
            writer.writerow([rho,
                             int(dictionary['rho'][rho]['avg proj num train']),
                             dictionary['rho'][rho]['unconverged rate train'],
                             int(dictionary['rho'][rho]['avg proj num val']),
                             dictionary['rho'][rho]['unconverged rate val'],
                             int(dictionary['rho'][rho]['avg proj num test']),
                             dictionary['rho'][rho]['unconverged rate test'],
                             int((dictionary['rho'][rho]['avg proj num train'] +
                                  dictionary['rho'][rho]['avg proj num val'] +
                                  dictionary['rho'][rho]['avg proj num test']) / 3),
                             (dictionary['rho'][rho]['unconverged rate train'] +
                              dictionary['rho'][rho]['unconverged rate val'] +
                              dictionary['rho'][rho]['unconverged rate test']) / 3])
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
