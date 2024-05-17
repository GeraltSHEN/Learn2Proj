import torch
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
    else:
        D1 = torch.eye(A.shape[0]).to(device)
        D2 = torch.eye(A.shape[1]).to(device)
    return D1, D2


def load_problem(args):
    if args.problem == 'primal_lp':
        A_primal = torch.load('./data/' + args.dataset + '/A_primal.pt').to(device)
        c_primal = torch.load('./data/' + args.dataset + '/c_primal.pt').to(device)

        D1, D2 = preconditioning(A_primal, args)
        scaled_A_primal = D1 @ A_primal @ D2
        scaled_c_primal = D2 @ c_primal

        problem = PrimalLP(scaled_c_primal, scaled_A_primal, args.primal_fx_idx, args.truncate_idx)
        return problem, D1.detach().cpu(), D2.detach().cpu()
    elif args.problem == 'dual_lp':
        A_dual = torch.load('./data/' + args.dataset + '/A_dual.pt').to(device)
        b_dual = torch.load('./data/' + args.dataset + '/b_dual.pt').to(device)

        D1, D2 = preconditioning(A_dual, args)
        scaled_A_dual = D1 @ A_dual @ D2
        scaled_b_dual = D1 @ b_dual

        problem = DualLP(scaled_b_dual, scaled_A_dual, args.dual_fx_idx, args.truncate_idx)
        return problem, D1.detach().cpu(), D2.detach().cpu()
    else:
        raise ValueError('Invalid problem')


# def load_dual_problem(args):
#     if args.problem == 'primal_lp':
#         A_dual = torch.load('./data/' + args.dataset + '/A_dual.pt').to(device)
#         b_dual = torch.load('./data/' + args.dataset + '/b_dual.pt').to(device)
#         problem = DualLP(b_dual, A_dual, args.dual_fx_idx, args.truncate_idx)
#         return problem
#     elif args.problem == 'dual_lp':
#         A_primal = torch.load('./data/' + args.dataset + '/A_primal.pt').to(device)
#         c_primal = torch.load('./data/' + args.dataset + '/c_primal.pt').to(device)
#         problem = PrimalLP(c_primal, A_primal, args.primal_fx_idx, args.truncate_idx)
#         return problem
#     else:
#         raise ValueError('Invalid problem')


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
    args.self_supervised = False
    args.batch_size = 1
    data = load_data(args)
    problem = load_problem(args)
    model = load_model(args, problem)
    assert problem.A.dtype == torch.float64
    # check if eq_residual and ineq_residual are correct
    for i, (inputs, targets) in enumerate(data['train']):
        inputs, targets = process_for_training(inputs, targets, args)
        # is float64?
        assert inputs.dtype == torch.float64
        assert targets.dtype == torch.float64
        eq_residual = problem.eq_residual(targets, inputs)
        ineq_residual = problem.ineq_residual(targets)
        assert eq_residual.dtype == torch.float64
        assert ineq_residual.dtype == torch.float64
        eq_max = torch.norm(eq_residual, float('inf'))
        ineq_max = torch.norm(ineq_residual, float('inf'))
        if eq_max > args.f_tol or ineq_max > args.f_tol:
            print('train bad point {: .0f}: eq max violation: {: .5f}; ineq max violation: {: .5f}'.format(i, eq_max, ineq_max))
    for i, (inputs, targets) in enumerate(data['val']):
        inputs, targets = process_for_training(inputs, targets, args)
        eq_residual = problem.eq_residual(targets, inputs)
        ineq_residual = problem.ineq_residual(targets)
        eq_max = torch.norm(eq_residual, float('inf'))
        ineq_max = torch.norm(ineq_residual, float('inf'))
        if eq_max > args.f_tol or ineq_max > args.f_tol:
            print('val bad point {: .0f}: eq max violation: {: .5f}; ineq max violation: {: .5f}'.format(i, eq_max, ineq_max))


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
        model = models.PrimalNN(input_dim=input_dim, hidden_dim=hidden_dim,
                                hidden_num=hidden_num, output_dim=output_dim,
                                truncate_idx=problem.truncate_idx, free_idx=problem.free_idx,
                                A=problem.A,
                                WzProj=Wz_proj, WbProj=Wb_proj,
                                max_iter=args.max_iter, f_tol=args.f_tol)
    elif args.model == 'dual_nn':
        Wz_proj, Wb_proj = load_W_proj(args, problem)
        model = models.DualNN(input_dim=input_dim, hidden_dim=hidden_dim,
                              hidden_num=hidden_num, output_dim=output_dim,
                              truncate_idx=problem.truncate_idx, free_idx=problem.free_idx,
                              A=problem.A, b=problem.b,
                              WzProj=Wz_proj, WbProj=Wb_proj,
                              max_iter=args.max_iter, f_tol=args.f_tol)
    elif args.model == 'vanilla_nn':
        model = models.VanillaNN(input_dim=input_dim, hidden_dim=hidden_dim,
                                 hidden_num=hidden_num, output_dim=output_dim,
                                 truncate_idx=problem.truncate_idx)
    elif args.model == 'dual_learn2proj':
        assert args.learn2proj
        Wz_proj, Wb_proj = load_W_proj(args, problem)
        model = models.DualLearn2Proj(input_dim=input_dim, hidden_dim=hidden_dim,
                                      hidden_num=hidden_num, output_dim=output_dim,
                                      proj_hidden_dim=args.proj_hidden_dim, proj_hidden_num=args.proj_hidden_num,
                                      truncate_idx=problem.truncate_idx, free_idx=problem.free_idx,
                                      A=problem.A, b=problem.b,
                                      WzProj=Wz_proj, WbProj=Wb_proj,
                                      max_iter=args.max_iter, f_tol=args.f_tol)

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




