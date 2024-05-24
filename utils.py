import torch
import csv
import json
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import models
from torch.utils.data import DataLoader, TensorDataset
from problem import *
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)


def calc_W_proj(A):
    start = time.time()
    chunk = torch.mm(A.t(), torch.inverse(torch.mm(A, A.t())))
    Wz_proj = torch.eye(A.shape[-1]).to(device) - torch.mm(chunk, A)
    Wb_proj = chunk
    end = time.time()
    print(f'W_proj calculation time: {end - start:.4f}s, A shape: {A.shape}')
    return Wz_proj, Wb_proj


def load_W_proj(args, problem):
    if 'primal' in args.problem:
        extension = 'primal'
    elif 'dual' in args.problem:
        extension = 'dual'
    else:
        raise ValueError('Invalid problem')
    Wz_proj, Wb_proj = calc_W_proj(problem.A)
    torch.save(Wz_proj.detach().cpu(), './data/' + args.dataset + f'/{extension}_Wz_proj.pt')
    torch.save(Wb_proj.detach().cpu(), './data/' + args.dataset + f'/{extension}_Wb_proj.pt')
    print(f'{extension}_Wz_proj.pt and {extension}_Wb_proj.pt saved. Terminating.')
    return Wz_proj, Wb_proj
    # try:
    #     Wz_proj = torch.load('./data/' + args.dataset + f'/{extension}_Wz_proj.pt').to(device)
    #     Wb_proj = torch.load('./data/' + args.dataset + f'/{extension}_Wb_proj.pt').to(device)
    #     return Wz_proj, Wb_proj
    # except:
    #     print(f'{extension}_Wz_proj.pt and {extension}_Wb_proj.pt not found. Calculating...')
    #     Wz_proj, Wb_proj = calc_W_proj(problem.A)
    #     torch.save(Wz_proj.detach().cpu(), './data/' + args.dataset + f'/{extension}_Wz_proj.pt')
    #     torch.save(Wb_proj.detach().cpu(), './data/' + args.dataset + f'/{extension}_Wb_proj.pt')
    #     print(f'{extension}_Wz_proj.pt and {extension}_Wb_proj.pt saved. Terminating.')
    #     return Wz_proj, Wb_proj


def preconditioning(A, args):
    if args.precondition == 'Pock-Chambolle':
        row_norms = torch.norm(A, dim=1, p=1)
        col_norms = torch.norm(A, dim=0, p=1)
        D1 = torch.diag(torch.sqrt(row_norms))
        D2 = torch.diag(torch.sqrt(col_norms))
    elif args.precondition == 'Ruiz':
        row_norms = torch.norm(A, dim=1, p=float('inf'))
        col_norms = torch.norm(A, dim=0, p=float('inf'))
        D1 = torch.diag(torch.sqrt(row_norms))
        D2 = torch.diag(torch.sqrt(col_norms))
    elif args.precondition == 'none':
        D1 = torch.eye(A.shape[0]).to(device)
        D2 = torch.eye(A.shape[1]).to(device)
    else:
        raise ValueError('Invalid precondition')
    if isinstance(args.f_tol, float):
        print(f'Scaling eq_tol by D1')
        scale_tolerance(D1, args)
        print(f'Scaled eq_tol range: ({args.eq_tol.min()} - {args.eq_tol.max()})')
    return D1, D2


def scale_tolerance(D1, args):
    assert isinstance(args.f_tol, float)
    eq_tolerance = args.f_tol * torch.ones(D1.shape[0]).to(device)
    # scale the tolerance by D1 @ tolerance
    scaled_eq_tolerance = D1 @ eq_tolerance
    args.eq_tol = scaled_eq_tolerance


def load_problem(args):
    if args.problem == 'primal_lp':
        A_primal = torch.load('./data/' + args.dataset + '/A_primal.pt').to(device)
        c_primal = torch.load('./data/' + args.dataset + '/c_primal.pt').to(device)

        D1, D2 = preconditioning(A_primal, args)
        scaled_A_primal = D1 @ A_primal @ D2
        scaled_c_primal = D2 @ c_primal

        problem = PrimalLP(scaled_c_primal, scaled_A_primal, args.primal_fx_idx, args.truncate_idx)
        print(f'A range: {A_primal.min()} - {A_primal.max()}')
        print(f'scaled_A range: {scaled_A_primal.min()} - {scaled_A_primal.max()}')
        return problem, D1.detach().cpu(), D2.detach().cpu()
    elif args.problem == 'dual_lp':
        A_dual = torch.load('./data/' + args.dataset + '/A_dual.pt').to(device)
        b_dual = torch.load('./data/' + args.dataset + '/b_dual.pt').to(device)

        D1, D2 = preconditioning(A_dual, args)
        scaled_A_dual = D1 @ A_dual @ D2
        scaled_b_dual = D1 @ b_dual

        problem = DualLP(scaled_b_dual, scaled_A_dual, args.dual_fx_idx, args.truncate_idx)
        print(f'A range: {A_dual.min()} - {A_dual.max()}')
        print(f'scaled_A range: {scaled_A_dual.min()} - {scaled_A_dual.max()}')
        return problem, D1.detach().cpu(), D2.detach().cpu()
    else:
        raise ValueError('Invalid problem')


def load_data(args):
    problem, D1, D2 = load_problem(args)

    input_train = torch.load('./data/' + args.dataset + '/train/input_train.pt')
    input_train = input_train @ D1.t()

    if args.self_supervised:
        extension = 'self_'
        target_train = torch.zeros(input_train.shape[0])
    else:
        extension = ''
        target_train = torch.load('./data/' + args.dataset + '/train/' + extension + 'target_train.pt')

    if args.test_val_train == 'test':
        input_val = torch.load('./data/' + args.dataset + '/test/input_test.pt')
        target_val = torch.load('./data/' + args.dataset + '/test/' + extension + 'target_test.pt')
    elif args.test_val_train == 'val':
        input_val = torch.load('./data/' + args.dataset + '/val/input_val.pt')
        target_val = torch.load('./data/' + args.dataset + '/val/' + extension + 'target_val.pt')
    elif args.test_val_train == 'train':
        input_val = input_train
        target_val = target_train
    else:
        raise ValueError('Invalid test_val_train')
    input_val = input_val @ D1.t()

    train_shape_in = input_train.shape
    train_shape_out = target_train.shape
    val_shape_in = input_val.shape
    val_shape_out = target_val.shape
    print(f'train_shape_in: {train_shape_in}\n'
          f'train_shape_out: {train_shape_out}\n'
          f'val_shape_in: {val_shape_in}\n'
          f'val_shape_out: {val_shape_out}')
    # mean, std
    mean = input_train.mean(dim=-1)
    std = input_train.std(dim=-1)

    train_data = TensorDataset(input_train, target_train)
    val_data = TensorDataset(input_val, target_val)
    train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    if args.test_val_train == 'test':
        val = DataLoader(val_data, batch_size=1, shuffle=False)
    else:
        val = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    data_dict = {'train': train, 'val': val,
                 'mean': mean, 'std': std,
                 'train_shape_in': train_shape_in, 'train_shape_out': train_shape_out,
                 'val_shape_in': val_shape_in, 'val_shape_out': val_shape_out}
    return data_dict, problem


def data_sanity_check(args):
    problem, D1, D2 = load_problem(args)
    assert problem.A.dtype == torch.float64

    input_train = torch.load('./data/' + args.dataset + '/train/input_train.pt')
    input_train = input_train @ D1.t()
    target_train = torch.load('./data/' + args.dataset + '/train/' + '' + 'target_train.pt')

    input_val = torch.load('./data/' + args.dataset + '/val/input_val.pt')
    input_val = input_val @ D1.t()
    target_val = torch.load('./data/' + args.dataset + '/val/' + '' + 'target_val.pt')

    input_test = torch.load('./data/' + args.dataset + '/test/input_test.pt')
    input_test = input_test @ D1.t()
    target_test = torch.load('./data/' + args.dataset + '/test/' + '' + 'target_test.pt')

    train_data = TensorDataset(input_train, target_train)
    val_data = TensorDataset(input_val, target_val)
    test_data = TensorDataset(input_test, target_test)
    train = DataLoader(train_data, batch_size=1, shuffle=False)
    val = DataLoader(val_data, batch_size=1, shuffle=False)
    test = DataLoader(test_data, batch_size=1, shuffle=False)

    def sanity_check(data, data_type):
        for i, (inputs, targets) in enumerate(data):
            inputs, targets = process_for_training(inputs, targets, args)
            assert inputs.dtype == torch.float64
            assert targets.dtype == torch.float64
            eq_residual = problem.eq_residual(targets, inputs)
            ineq_residual = problem.ineq_residual(targets)
            assert eq_residual.dtype == torch.float64
            assert ineq_residual.dtype == torch.float64
            eq_stopping_criterion = torch.mean(torch.abs(eq_residual), dim=0)
            ineq_stopping_criterion = torch.mean(torch.abs(ineq_residual), dim=0)
            # eq_max = torch.norm(eq_residual, float('inf'))
            # ineq_max = torch.norm(ineq_residual, float('inf'))
            if (eq_stopping_criterion > args.eq_tol).all() or (ineq_stopping_criterion > args.ineq_tol).all():
                print('{} bad point {: .0f}: eq max violation: {: .5f}; ineq max violation: {: .5f}'.format(
                    data_type, i, eq_stopping_criterion.max(), ineq_stopping_criterion.max()))
    sanity_check(train, 'train')
    sanity_check(val, 'val')
    sanity_check(test, 'test')


def projection_sanity_check(args):
    problem, D1, D2 = load_problem(args)
    assert problem.A.dtype == torch.float64

    input_train = torch.load('./data/' + args.dataset + '/train/input_train.pt')
    input_train = input_train @ D1.t()
    target_train = torch.load('./data/' + args.dataset + '/train/' + '' + 'target_train.pt')

    input_val = torch.load('./data/' + args.dataset + '/val/input_val.pt')
    input_val = input_val @ D1.t()
    target_val = torch.load('./data/' + args.dataset + '/val/' + '' + 'target_val.pt')

    input_test = torch.load('./data/' + args.dataset + '/test/input_test.pt')
    input_test = input_test @ D1.t()
    target_test = torch.load('./data/' + args.dataset + '/test/' + '' + 'target_test.pt')

    train_data = TensorDataset(input_train, target_train)
    val_data = TensorDataset(input_val, target_val)
    test_data = TensorDataset(input_test, target_test)
    train = DataLoader(train_data, batch_size=1, shuffle=False)
    val = DataLoader(val_data, batch_size=1, shuffle=False)
    test = DataLoader(test_data, batch_size=1, shuffle=False)

    def load_solvers(args, problem):
        Wz_proj, Wb_proj = load_W_proj(args, problem)
        pocs_solver = models.POCS(problem.free_idx, problem.A, Wz_proj, args.max_iter, args.eq_tol, args.ineq_tol)
        eapm_solver = models.EAPM(problem.free_idx, problem.A, Wz_proj, args.max_iter, args.eq_tol, args.ineq_tol, args.rho)
        return pocs_solver, eapm_solver, Wb_proj.t().requires_grad_(False).to(device)

    pocs_solver, eapm_solver, Wb_proj = load_solvers(args, problem)

    def sanity_check(data, solver, Wb_proj, args):
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

    def run_expr(train, val, test, pocs_solver, eapm_solver, Wb_proj, args):
        print(f'-----{args.precondition} projection check, '
              f'eq_tol: ({args.eq_tol.min()} - {args.eq_tol.max()}) '
              f'ineq_tol: {args.ineq_tol} '
              f'max_iter: {args.max_iter} -----')
        print(f'----- 1. train pocs')
        avg_proj_num_train_pocs, unconverged_rate_train_pocs, unconverged_idx_train_pocs = sanity_check(train,
                                                                                                        pocs_solver,
                                                                                                        Wb_proj, args)
        print(f'----- 2. val pocs')
        avg_proj_num_val_pocs, unconverged_rate_val_pocs, unconverged_idx_val_pocs = sanity_check(val,
                                                                                                  pocs_solver,
                                                                                                  Wb_proj, args)
        print(f'----- 3. test pocs')
        avg_proj_num_test_pocs, unconverged_rate_test_pocs, unconverged_idx_test_pocs = sanity_check(test,
                                                                                                     pocs_solver,
                                                                                                     Wb_proj, args)
        print(f'----- 4. train eapm')
        avg_proj_num_train_eapm, unconverged_rate_train_eapm, unconverged_idx_train_eapm = sanity_check(train,
                                                                                                        eapm_solver,
                                                                                                        Wb_proj, args)
        print(f'----- 5. val eapm')
        avg_proj_num_val_eapm, unconverged_rate_val_eapm, unconverged_idx_val_eapm = sanity_check(val,
                                                                                                  eapm_solver,
                                                                                                  Wb_proj, args)
        print(f'----- 6. test eapm')
        avg_proj_num_test_eapm, unconverged_rate_test_eapm, unconverged_idx_test_eapm = sanity_check(test,
                                                                                                     eapm_solver,
                                                                                                     Wb_proj, args)
        dictionary = {'dataset': args.dataset,
                      'precond': args.precondition,
                      'eq_tol': args.eq_tol,
                      'ineq_tol': args.ineq_tol,
                      'max_iter': args.max_iter,
                      'rho': args.rho,
                      'avg proj num train pocs': avg_proj_num_train_pocs,
                      'avg proj num train eapm': avg_proj_num_train_eapm,
                      'unconverged rate train pocs': unconverged_rate_train_pocs,
                      'unconverged rate train eapm': unconverged_rate_train_eapm,
                      'avg proj num val pocs': avg_proj_num_val_pocs,
                      'avg proj num val eapm': avg_proj_num_val_eapm,
                      'unconverged rate val pocs': unconverged_rate_val_pocs,
                      'unconverged rate val eapm': unconverged_rate_val_eapm,
                      'avg proj num test pocs': avg_proj_num_test_pocs,
                      'avg proj num test eapm': avg_proj_num_test_eapm,
                      'unconverged rate test pocs': unconverged_rate_test_pocs,
                      'unconverged rate test eapm': unconverged_rate_test_eapm}
        w = csv.writer(open('./data/sanity_check/' + args.dataset + '_' + args.precondition + str(args.rho) + '.csv', 'w'))
        for key, val in dictionary.items():
            w.writerow([key, val])

        unconverged_idx = {'unconverged_idx_train_pocs': unconverged_idx_train_pocs,
                           'unconverged_idx_val_pocs': unconverged_idx_val_pocs,
                           'unconverged_idx_test_pocs': unconverged_idx_test_pocs,
                           'unconverged_idx_train_eapm': unconverged_idx_train_eapm,
                           'unconverged_idx_val_eapm': unconverged_idx_val_eapm,
                           'unconverged_idx_test_eapm': unconverged_idx_test_eapm}
        with open('./data/sanity_check/' + args.dataset + '_' + args.precondition + str(args.rho) + '.json', 'w') as f:
            json.dump(unconverged_idx, f)

    run_expr(train, val, test, pocs_solver, eapm_solver, Wb_proj, args)


def load_model(args, problem):
    input_dim = problem.truncate_idx[1] - problem.truncate_idx[0]
    output_dim = problem.var_num
    if 'primal' in problem.name:
        hidden_dim = args.primal_hidden_dim
        hidden_num = args.primal_hidden_num
    elif 'dual' in problem.name:
        hidden_dim = args.dual_hidden_dim
        hidden_num = args.dual_hidden_num
    else:
        raise ValueError('Invalid problem')

    if args.model == 'primal_nn':
        Wz_proj, Wb_proj = load_W_proj(args, problem)
        model = models.OptProjNN(input_dim=input_dim, hidden_dim=hidden_dim,
                                 hidden_num=hidden_num, output_dim=output_dim,
                                 truncate_idx=problem.truncate_idx, free_idx=problem.free_idx,
                                 A=problem.A,
                                 WzProj=Wz_proj, WbProj=Wb_proj,
                                 max_iter=args.max_iter, eq_tol=args.eq_tol, ineq_tol= args.ineq_tol,
                                 projection=args.projection, rho=args.rho)

        # if args.projection == 'POCS':
        #     model = models.PrimalNN(input_dim=input_dim, hidden_dim=hidden_dim,
        #                             hidden_num=hidden_num, output_dim=output_dim,
        #                             truncate_idx=problem.truncate_idx, free_idx=problem.free_idx,
        #                             A=problem.A,
        #                             WzProj=Wz_proj, WbProj=Wb_proj,
        #                             max_iter=args.max_iter, f_tol=args.f_tol)
        # elif args.projection == 'EAPM':
        #     model = models.PrimalEAPM(input_dim=input_dim, hidden_dim=hidden_dim,
        #                               hidden_num=hidden_num, output_dim=output_dim,
        #                               truncate_idx=problem.truncate_idx, free_idx=problem.free_idx,
        #                               WzProj=Wz_proj, WbProj=Wb_proj,
        #                               rho=args.rho, max_iter=args.max_iter, f_tol=args.f_tol, A=problem.A)

    elif args.model == 'dual_nn':
        Wz_proj, Wb_proj = load_W_proj(args, problem)
        model = models.DualOptProjNN(input_dim=input_dim, hidden_dim=hidden_dim,
                                     hidden_num=hidden_num, output_dim=output_dim,
                                     truncate_idx=problem.truncate_idx, free_idx=problem.free_idx,
                                     A=problem.A,
                                     WzProj=Wz_proj, WbProj=Wb_proj,
                                     max_iter=args.max_iter, eq_tol=args.eq_tol, ineq_tol= args.ineq_tol,
                                     projection=args.projection, rho=args.rho,
                                     b_dual=problem.b)

        # if args.projection == 'POCS':
        #     model = models.DualNN(input_dim=input_dim, hidden_dim=hidden_dim,
        #                           hidden_num=hidden_num, output_dim=output_dim,
        #                           truncate_idx=problem.truncate_idx, free_idx=problem.free_idx,
        #                           A=problem.A, b=problem.b,
        #                           WzProj=Wz_proj, WbProj=Wb_proj,
        #                           max_iter=args.max_iter, f_tol=args.f_tol)
        # elif args.projection == 'EAPM':
        #     model = models.DualEAPM(input_dim=input_dim, hidden_dim=hidden_dim,
        #                               hidden_num=hidden_num, output_dim=output_dim,
        #                               truncate_idx=problem.truncate_idx, free_idx=problem.free_idx,
        #                               WzProj=Wz_proj, WbProj=Wb_proj,
        #                               rho=args.rho, max_iter=args.max_iter, f_tol=args.f_tol)

    elif args.model == 'vanilla_nn':
        model = models.VanillaNN(input_dim=input_dim, hidden_dim=hidden_dim,
                                 hidden_num=hidden_num, output_dim=output_dim,
                                 truncate_idx=problem.truncate_idx)
    # elif args.model == 'dual_learn2proj':
    #     assert args.learn2proj
    #     Wz_proj, Wb_proj = load_W_proj(args, problem)
    #     model = models.DualLearn2Proj(input_dim=input_dim, hidden_dim=hidden_dim,
    #                                   hidden_num=hidden_num, output_dim=output_dim,
    #                                   proj_hidden_dim=args.proj_hidden_dim, proj_hidden_num=args.proj_hidden_num,
    #                                   truncate_idx=problem.truncate_idx, free_idx=problem.free_idx,
    #                                   A=problem.A, b=problem.b,
    #                                   WzProj=Wz_proj, WbProj=Wb_proj,
    #                                   max_iter=args.max_iter, f_tol=args.f_tol)

    else:
        raise ValueError('Invalid model')
    model = model.double().to(device)
    return model


def get_optimizer(args, model, proj=False):
    if proj:
        params = model.projection_layers.parameters()
        print(f'proj_optimizer for projection_layers')
        # params = list(model[0].parameters()) + list(model[1].parameters())
    else:
        if args.learn2proj:
            params = model.optimality_layers.parameters()
            print(f'optimizer for optimality_layers')
        else:
            params = model.parameters()
            print(f'optimizer for all layers')

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError('Invalid optimizer')
    return optimizer


def get_optimizer_new(args, model):
    # Instead of having just *one* optimizer, we will have a ``dict`` of optimizers
    # for every parameter so we could reference them in our hook.
    if args.optimizer == 'Adam':
        optimizer_dict = {p: torch.optim.Adam([p], foreach=False) for p in model.parameters()}
    elif args.optimizer == 'SGD':
        optimizer_dict = {p: torch.optim.SGD([p], foreach=False) for p in model.parameters()}

    # Define our hook, which will call the optimizer ``step()`` and ``zero_grad()``
    def optimizer_hook(parameter) -> None:
        optimizer_dict[parameter].step()
        optimizer_dict[parameter].zero_grad()

    # Register the hook onto every parameter
    for p in model.parameters():
        p.register_post_accumulate_grad_hook(optimizer_hook)
    print(f'register_post_accumulate_grad_hook to save gradient memory')

    # Now remember our previous ``train()`` function? Since the optimizer has been
    # fused into the backward, we can remove the optimizer step and zero_grad calls.
    #     # call our forward and backward
    #     loss = model.forward(fake_image)
    #     loss.sum().backward()
        # optimizer update --> no longer needed!
        # optimizer.step()
        # optimizer.zero_grad()


def get_violation_mean_max_worst(residual):
    # mean = torch.abs(residual).mean()  # mean of constraints and samples
    mean = torch.norm(residual, p=2, dim=-1).mean()  # mean of constraints and samples
    max = torch.norm(residual, float('inf'), dim=-1).mean()  # mean of samples, max of constraints
    worst = torch.norm(residual, float('inf'))  # max of samples and constraints
    return mean, max, worst


def get_scale_mean_max_worst(rhs):
    mean = 1 + torch.norm(rhs, p=2, dim=-1).mean()  # mean of rhs and samples
    max = 1 + torch.norm(rhs, float('inf'), dim=-1).mean()  # mean of samples, max of rhs
    worst = 1 + torch.norm(rhs, float('inf'))  # max of samples and rhs
    return mean, max, worst


def get_scaled_violation_mean_max_worst(residual, rhs):
    mean, max, worst = get_violation_mean_max_worst(residual)
    scale_mean, scale_max, scale_worst = get_scale_mean_max_worst(rhs)
    return mean / scale_mean, max / scale_max, worst / scale_worst


def get_gap_mean_worst(gap):
    mean = torch.abs(gap).mean()  # mean of samples
    worst = torch.norm(gap, float('inf'))  # max of samples
    return mean, worst


def get_loss(z_star, z1, targets, inputs, problem, args, loss_type):
    predicted_obj = problem.obj_fn(z_star, inputs)
    if loss_type == 'obj':
        return predicted_obj.mean()
    elif loss_type == 'soft':
        eq_residual = problem.eq_residual(z1, inputs)
        ineq_residual = problem.ineq_residual(z1)
        penalty = args.penalty_g * eq_residual.pow(2).sum(dim=-1) + args.penalty_h * ineq_residual.pow(2).sum(dim=-1)
        return (predicted_obj + penalty).mean()
    elif loss_type == 'mse':
        return F.mse_loss(z_star, targets)
    else:
        raise ValueError('Invalid loss_type')


def process_for_training(inputs, targets, args):
    inputs = inputs.to(device)
    targets = targets.to(device)
    return inputs, targets




