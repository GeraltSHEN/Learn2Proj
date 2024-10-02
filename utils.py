import torch
import csv
import json
import numpy as np
from scipy.linalg import null_space
import torch.optim as optim
import torch.nn.functional as F
import models
from torch.utils.data import DataLoader, TensorDataset
from problem import *
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#torch.set_default_dtype(torch.float64)


def print_sparsity(A):
    assert not A.is_sparse
    print(f'tensor sparsity: {(1 - torch.count_nonzero(A) / A.nelement()) * 100:.2f} %')


def print_memory(A):
    if A.is_sparse:
        # Calculate the memory usage
        # Memory usage = nnz * element size + (2 * nnz + n) * int size
        nnz = A._nnz()  # Number of non-zero elements
        element_size = A.element_size()  # Size of one element in bytes
        index_size = A.indices().element_size()  # Size of one index in bytes
        memory_usage = nnz * element_size + (2 * nnz) * index_size
        print(f'tensor is sparse, memory: {memory_usage / 1024 / 1024 / 1024:.2f} GB')
        print(f'if dense, memory: {A.element_size() * A.nelement() / 1024 / 1024 / 1024:.2f} GB')
    else:
        print(f'tensor is dense, memory: {A.element_size() * A.nelement() / 1024 / 1024 / 1024:.2f} GB')
        print_sparsity(A)


def adjust_precision(args, tensor, tensor_name=''):
    print(tensor_name + 'tensor dtype:', tensor.dtype)
    if not args.float64:
        if tensor.dtype == torch.float64:
            print(tensor_name + 'tensor new dtype: float32')
            tensor = tensor.float()
    else:
        if tensor.dtype == torch.float32:
            print(tensor_name + 'tensor new dtype: float64')
            tensor = tensor.double()
    return tensor


def load_fixed_params(args):
    if args.problem == 'primal_lp':
        A_TYPE = "A_primal"
        bc_TYPE = "c_primal"
    elif args.problem == 'dual_lp':
        A_TYPE = "A_dual"
        bc_TYPE = "b_dual"
    else:
        raise ValueError('Invalid problem')
    A = torch.load(f'./data/{args.dataset}/{A_TYPE}_dense.pt')  # here we change it to dense
    b_or_c = torch.load(f'./data/{args.dataset}/{bc_TYPE}.pt')
    A = adjust_precision(args, A, 'A_')
    b_or_c = adjust_precision(args, b_or_c, 'b_or_c_')

    print(f'{A_TYPE} shape: {A.shape}, {bc_TYPE} shape: {b_or_c.shape}')
    # if A.is_sparse:
    #     print(f'sparse {A_TYPE} (loaded) memory: {A.element_size() * A._nnz() / 1024 / 1024 / 1024:.2f} GB')
    #     print(f'dense {A_TYPE} memory: {A.element_size() * A.nelement() / 1024 / 1024 / 1024:.2f} GB')
    # else:
    #     print(f'dense {A_TYPE} (loaded) memory: {A.element_size() * A.nelement() / 1024 / 1024 / 1024:.2f} GB')
    #     print(f'{A_TYPE} sparsity: {1 - torch.count_nonzero(A) / A.nelement()}')
    #     A = A.to_sparse()
    #     print(f'sparse {A_TYPE} (reloaded) memory: {A.element_size() * A._nnz() / 1024 / 1024 / 1024:.2f} GB')
    #     torch.save(A, f'./data/{args.dataset}/{A_TYPE}.pt')
    #     print(f'original dense {A_TYPE} is replaced and saved as sparse')
    return A, b_or_c


def load_unfixed_params(args, DATA_TYPE, for_training=False):
    assert args.self_supervised
    inputs = torch.load(f'./data/{args.dataset}/{DATA_TYPE}/input_{DATA_TYPE}.pt')
    inputs = adjust_precision(args, inputs, 'inputs_')

    if for_training:
        targets = torch.zeros(inputs.shape[0])
    else:
        targets = torch.load(f'./data/{args.dataset}/{DATA_TYPE}/self_target_{DATA_TYPE}.pt')

    targets = adjust_precision(args, targets, 'targets_')
    print(f'input_{DATA_TYPE} shape: {inputs.shape}, target_{DATA_TYPE} shape: {targets.shape}')
    return inputs, targets


def load_preconditions(args, A):
    if args.precondition == 'none':
        if 'primal' in args.problem:
            D1 = torch.ones(args.primal_const_num)
            D2 = torch.ones(args.primal_var_num)
        elif 'dual' in args.problem:
            D1 = torch.ones(args.dual_const_num)
            D2 = torch.ones(args.dual_var_num)
        else:
            raise ValueError('Invalid problem')

    elif args.precondition == 'Pock-Chambolle':
        row_norms = torch.norm(A, dim=1, p=1)
        col_norms = torch.norm(A, dim=0, p=1)
        D1 = torch.sqrt(row_norms)
        D2 = torch.sqrt(col_norms)
    elif args.precondition == 'Ruiz':
        row_norms = torch.norm(A, dim=1, p=float('inf'))
        col_norms = torch.norm(A, dim=0, p=float('inf'))
        D1 = torch.sqrt(row_norms)
        D2 = torch.sqrt(col_norms)
    else:
        raise ValueError('Invalid precondition')
    D1 = adjust_precision(args, D1, 'D1_')
    D2 = adjust_precision(args, D2, 'D2_')
    print(f'D1 shape: {D1.shape}, D2 shape: {D2.shape}')
    if isinstance(args.f_tol, float):
        print('===' * 10)
        print(f'Scaling eq_tol by D1')
        print('ineq_tol is not scaled because rhs is all 0')
        scaled_eq_tolerance = args.f_tol * D1
        args.eq_tol = scaled_eq_tolerance
        args.ineq_tol = args.f_tol
        print(f'Scaled eq_tol range: [{args.eq_tol.min()}, {args.eq_tol.max()}]')

    # else:
    #     #
    #     D1 = torch.load(f'./data/{args.dataset}/{args.precondition}_D1.pt')
    #     D2 = torch.load(f'./data/{args.dataset}/{args.precondition}_D2.pt')

    # scale tolerance
    # print('==='*10)
    # print('Scaling eq_tol by D1')
    # print('ineq_tol is not scaled because rhs is all 0')
    # scaled_eq_tolerance = args.f_tol * D1
    # args.eq_tol = scaled_eq_tolerance
    # args.ineq_tol = args.f_tol

    return D1, D2


def load_problem_new(args):
    A, b_or_c = load_fixed_params(args)
    D1, D2 = load_preconditions(args, A)
    scaled_A = D1[:, None] * A * D2
    scaled_b_or_c = D2 * b_or_c
    if 'primal' in args.problem:
        problem = PrimalLP(scaled_b_or_c.to(device), scaled_A.to(device), args.primal_fx_idx, args.truncate_idx)
    elif 'dual' in args.problem:
        problem = DualLP(scaled_b_or_c.to(device), scaled_A.to(device), args.dual_fx_idx, args.truncate_idx)
    else:
        raise ValueError('Invalid problem')
    print('==='*10)
    print(f'{args.problem} problem loaded')

    def get_range(m):
        if m.is_sparse:
            m = m.to_dense()
        m = m.flatten()
        abs_values = torch.abs(m[m != 0])
        return abs_values.min(), abs_values.max()
    Al, Au = get_range(A)
    scaled_Al, scaled_Au = get_range(scaled_A)
    print(f'A range: [{Al}, {Au}]')
    print(f'scaled_A range: [{scaled_Al}, {scaled_Au}]')

    problem.D1 = D1
    problem.D2 = D2

    return problem


def load_data_new(args, problem):
    if args.job in ['training']:
        input_train, target_train = load_unfixed_params(args, 'train', True)
        input_val, target_val = load_unfixed_params(args, 'val', False)
        input_test, target_test = load_unfixed_params(args, 'test', False)
        input_train = input_train * problem.D1
        input_val = input_val * problem.D1
        input_test = input_test * problem.D1

        problem.obj_example = (target_val.sum() + target_test.sum()) / (len(target_val) + len(target_test))  #todo: try tuning this
        print(f'example objective value: {problem.obj_example}')

        train_data = TensorDataset(input_train, target_train)
        val_data = TensorDataset(input_val, target_val)
        test_data = TensorDataset(input_test, target_test)
        train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
        test = DataLoader(test_data, batch_size=1, shuffle=False)

        data = {'train': train, 'val': val, 'test': test}
        return data

    elif args.job in ['baseline_pocs', 'run_proj_exp', 'rho_search']:
        print(f'Loading data for {args.job}')
        input_train, target_train = load_unfixed_params(args, 'train', False)
        input_val, target_val = load_unfixed_params(args, 'val', False)
        input_test, target_test = load_unfixed_params(args, 'test', False)
        input_train = input_train * problem.D1
        input_val = input_val * problem.D1
        input_test = input_test * problem.D1

        # TODO: remember to change this back to full dataset
        train_data = TensorDataset(input_train, target_train)
        val_data = TensorDataset(input_val, target_val)
        test_data = TensorDataset(input_test, target_test)
        train = DataLoader(train_data, batch_size=1, shuffle=False)
        val = DataLoader(val_data, batch_size=1, shuffle=False)
        test = DataLoader(test_data, batch_size=1, shuffle=False)

        data = {'train': train, 'val': val, 'test': test}
        return data


def load_model_new(args, problem):
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
        Q, z0 = load_LDR(args, problem)
        if args.projection == 'LDRPM' and args.precondition != 'none':
            raise ValueError('LDRPM does not support preconditioning, because Q and z0 in LDR are not scaled')

        model = models.OptProjNN(input_dim=input_dim, hidden_dim=hidden_dim,
                                 hidden_num=hidden_num, output_dim=output_dim,
                                 truncate_idx=problem.truncate_idx, free_idx=problem.free_idx,
                                 A=problem.A, Q=Q, z0=z0,
                                 WzProj=Wz_proj, WbProj=Wb_proj,
                                 max_iter=args.max_iter, eq_tol=args.eq_tol, ineq_tol=args.ineq_tol,
                                 proj_method=args.projection, rho=args.rho)

    elif args.model == 'dual_nn':
        Wz_proj, Wb_proj = load_W_proj(args, problem)
        model = models.DualOptProjNN(input_dim=input_dim, hidden_dim=hidden_dim,
                                     hidden_num=hidden_num, output_dim=output_dim,
                                     truncate_idx=problem.truncate_idx, free_idx=problem.free_idx,
                                     A=problem.A,
                                     WzProj=Wz_proj, WbProj=Wb_proj,
                                     max_iter=args.max_iter, eq_tol=args.eq_tol, ineq_tol=args.ineq_tol,
                                     projection=args.projection, rho=args.rho,
                                     b_dual=problem.b)

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

    model = model.to(device)
    return model







# def calc_QR_proj(A):
#     # warning: this function will be deprecated
#     start = time.time()
#     if A.is_sparse:
#         Q_proj, R_proj = torch.linalg.qr(A.t().to_dense(), mode='reduced')
#     else:
#         Q_proj, R_proj = torch.linalg.qr(A.t(), mode='reduced')
#     R_proj = torch.inverse((R_proj.t()))
#     end = time.time()
#     print(f'A transpose shape: {A.t().shape}')
#     print(f'Q_proj shape: {Q_proj.shape}, R_proj shape: {R_proj.shape}')
#     print(f'QR_proj calculation time: {end - start:.4f}s')
#
#     print('==='*10)
#     print('Q_proj and R_proj in dense')
#     print_memory(Q_proj)
#     print_memory(R_proj)
#
#     # usually it is not worth it to convert Q_proj and R_proj to sparse
#     # print('==='*10)
#     # print('transform Q_proj and R_proj to sparse')
#     # Q_proj = Q_proj.to_sparse()
#     # R_proj = R_proj.to_sparse()
#     # print_memory(Q_proj)
#     # print_memory(R_proj)
#
#     return Q_proj, R_proj


# def load_QR_proj(args, problem):
#     # warning: this function will be deprecated
#     if 'primal' in args.problem:
#         extension = 'primal'
#     elif 'dual' in args.problem:
#         extension = 'dual'
#     else:
#         raise ValueError('Invalid problem')
#     try:
#         Q_proj = torch.load('./data/' + args.dataset + f'/{extension}_Q_proj.pt').to(device)
#         R_proj = torch.load('./data/' + args.dataset + f'/{extension}_R_proj.pt').to(device)
#         return Q_proj, R_proj
#     except:
#         print(f'{extension}_Q_proj.pt and {extension}_R_proj.pt not found. Calculating...')
#         Q_proj, R_proj = calc_QR_proj(problem.A)
#         torch.save(Q_proj.detach().cpu(), './data/' + args.dataset + f'/{extension}_Q_proj.pt')
#         torch.save(R_proj.detach().cpu(), './data/' + args.dataset + f'/{extension}_R_proj.pt')
#         print(f'{extension}_Q_proj.pt and {extension}_R_proj.pt saved. Terminating.')
#         return Q_proj, R_proj


# def calc_chunk_proj(A):
#     # warning: this function will be deprecated
#     assert A.is_sparse
#     start = time.time()
#     AAT = torch.sparse.mm(A, A.t())
#     chunk = torch.inverse(AAT.to_dense())
#     end = time.time()
#     print(f'chunk_proj calculation time: {end - start:.4f}s')
#     print('==='*10)
#     print('chunk_proj in dense')
#     print_memory(chunk)
#     return chunk


# def load_chunk_proj(args, problem):
#     # warning: this function will be deprecated
#     if 'primal' in args.problem:
#         extension = 'primal'
#     elif 'dual' in args.problem:
#         extension = 'dual'
#     else:
#         raise ValueError('Invalid problem')
#     try:
#         chunk = torch.load('./data/' + args.dataset + f'/{extension}_chunk.pt').to(device)
#         return chunk
#     except:
#         print(f'{extension}_chunk.pt not found. Calculating...')
#         chunk = calc_chunk_proj(problem.A)
#         torch.save(chunk.detach().cpu(), './data/' + args.dataset + f'/{extension}_chunk.pt')
#         print(f'{extension}_chunk.pt calculated and saved.')
#         return chunk


# def calc_L_proj(args, problem):
#     # warning: this function will be
#     start = time.time()
#     chunk = load_chunk_proj(args, problem)
#     L = torch.linalg.cholesky(chunk)
#     end = time.time()
#     print(f'L_proj calculation time: {end - start:.4f}s')
#     print('===' * 10)
#     print('L_proj in dense')
#     print_memory(L)
#     return L


# def load_L_proj(args, problem):
#     # warning: this function will be deprecated
#     if 'primal' in args.problem:
#         extension = 'primal'
#     elif 'dual' in args.problem:
#         extension = 'dual'
#     else:
#         raise ValueError('Invalid problem')
#     try:
#         L = torch.load('./data/' + args.dataset + f'/{extension}_L_proj.pt').to(device)
#         return L
#     except:
#         print(f'{extension}_L_proj.pt not found. Calculating...')
#         L = calc_L_proj(args, problem)
#         torch.save(L.detach().cpu(), './data/' + args.dataset + f'/{extension}_L_proj.pt')
#         print(f'{extension}_L_proj.pt calculated and saved.')
#         return L


def calc_N(A):
    """
    return the orthonormal basis of the null space of A
    """
    # if A is in sparse format, change it into dense
    if A.is_sparse:
        A = A.to_dense()
    start = time.time()
    N = torch.from_numpy(null_space(A.detach().cpu().numpy())).to(device)
    end = time.time()
    print(f'N calculation time: {end - start:.4f}s, N shape: {N.shape}')
    print_memory(N)
    return N


def load_N(args, problem):
    if 'primal' in args.problem:
        extension = 'primal'
    elif 'dual' in args.problem:
        extension = 'dual'
    else:
        raise ValueError('Invalid problem')
    try:
        N = torch.load('./data/' + args.dataset + f'/{extension}_N.pt').to(device)
        return N
    except:
        print(f'{extension}_N.pt not found. Calculating...')
        N = calc_N(problem.A)
        torch.save(N.detach().cpu(), './data/' + args.dataset + f'/{extension}_N.pt')
        print(f'{extension}_N.pt saved. Terminating.')
        return N


def calc_W_proj(args, A):
    # warning: this function will be deprecated
    start = time.time()

    PD = torch.mm(A, A.t())

    try:
        chunk = torch.mm(A.t(), torch.inverse(PD))
    except:
        A = A.to_dense()
        chunk = torch.mm(A.t(), torch.inverse(PD))

    Wz_proj = torch.eye(A.shape[-1]).to(device) - torch.mm(chunk, A)
    Wb_proj = chunk
    end = time.time()
    print(f'W_proj calculation time: {end - start:.4f}s, A shape: {A.shape}')
    return Wz_proj, Wb_proj


def load_W_proj(args, problem):
    # warning: this function will be deprecated
    if 'primal' in args.problem:
        extension = 'primal'
    elif 'dual' in args.problem:
        extension = 'dual'
    else:
        raise ValueError('Invalid problem')
    try:
        Wz_proj = torch.load('./data/' + args.dataset + f'/{extension}_Wz_proj_{args.float64}_{args.precondition}.pt').to(device)
        Wb_proj = torch.load('./data/' + args.dataset + f'/{extension}_Wb_proj_{args.float64}_{args.precondition}.pt').to(device)
        Wz_proj = adjust_precision(args, Wz_proj, 'Wz_proj_')
        Wb_proj = adjust_precision(args, Wb_proj, 'Wb_proj_')
        return Wz_proj, Wb_proj
    except:
        print(f'{extension}_Wz_proj_{args.float64}_{args.precondition}.pt and {extension}_Wb_proj_{args.float64}_{args.precondition}.pt not found. Calculating...')
        Wz_proj, Wb_proj = calc_W_proj(args, problem.A)
        Wz_proj = adjust_precision(args, Wz_proj, 'Wz_proj_')
        Wb_proj = adjust_precision(args, Wb_proj, 'Wb_proj_')
        torch.save(Wz_proj.detach().cpu(), './data/' + args.dataset + f'/{extension}_Wz_proj_{args.float64}_{args.precondition}.pt')
        torch.save(Wb_proj.detach().cpu(), './data/' + args.dataset + f'/{extension}_Wb_proj_{args.float64}_{args.precondition}.pt')
        print(f'{extension}_Wz_proj.pt and {extension}_Wb_proj.pt saved. Terminating.')
        return Wz_proj, Wb_proj


def load_LDR(args, problem):
    if args.ldr_type == 'feas':
        Q_LDR = torch.load('./data/' + args.dataset + f'/Q_LDR.pt').to(device)
        z0_LDR = torch.load('./data/' + args.dataset + f'/z0_LDR.pt').to(device)
    elif args.ldr_type == 'opt':
        Q_LDR = torch.load('./data/' + args.dataset + f'/Q_opt_LDR.pt').to(device)
        z0_LDR = torch.load('./data/' + args.dataset + f'/z0_opt_LDR.pt').to(device)
    else:
        raise ValueError('Invalid ldr_type')
    Q_LDR = adjust_precision(args, Q_LDR, 'Q_LDR_')
    z0_LDR = adjust_precision(args, z0_LDR, 'z0_LDR_')
    return Q_LDR, z0_LDR


# def preconditioning(A, args):
#     if args.precondition == 'Pock-Chambolle':
#         row_norms = torch.norm(A, dim=1, p=1)
#         col_norms = torch.norm(A, dim=0, p=1)
#         D1 = torch.diag(torch.sqrt(row_norms))
#         D2 = torch.diag(torch.sqrt(col_norms))
#     elif args.precondition == 'Ruiz':
#         row_norms = torch.norm(A, dim=1, p=float('inf'))
#         col_norms = torch.norm(A, dim=0, p=float('inf'))
#         D1 = torch.diag(torch.sqrt(row_norms))
#         D2 = torch.diag(torch.sqrt(col_norms))
#     elif args.precondition == 'none':
#         D1 = torch.ones(A.shape[0]).to(device)
#         D2 = torch.ones(A.shape[1]).to(device)
#     else:
#         raise ValueError('Invalid precondition')
#     if isinstance(args.f_tol, float):
#         print(f'Scaling eq_tol by D1')
#         scale_tolerance(D1, args)
#         print(f'Scaled eq_tol range: [{args.eq_tol.min()}, {args.eq_tol.max()}]')
#     return D1, D2
#
#
# def scale_tolerance(D1, args):
#     scaled_eq_tolerance = args.f_tol * D1
#     args.eq_tol = scaled_eq_tolerance
#
#
# def load_problem(args):
#     if args.problem == 'primal_lp':
#         A_primal = torch.load('./data/' + args.dataset + '/A_primal.pt').to(device)
#         c_primal = torch.load('./data/' + args.dataset + '/c_primal.pt').to(device)
#
#         D1, D2 = preconditioning(A_primal, args)
#         # scaled_A_primal = D1 @ A_primal @ D2
#         # scaled_c_primal = D2 @ c_primal
#         # D1 D2 are vec(diagonal)
#         scaled_A_primal = D1[:, None] * A_primal * D2
#         scaled_c_primal = D2 * c_primal
#
#         problem = PrimalLP(scaled_c_primal, scaled_A_primal, args.primal_fx_idx, args.truncate_idx)
#         print(f'A range: {A_primal.min()} - {A_primal.max()}')
#         print(f'scaled_A range: {scaled_A_primal.min()} - {scaled_A_primal.max()}')
#         return problem, D1.detach().cpu(), D2.detach().cpu()
#     elif args.problem == 'dual_lp':
#         A_dual = torch.load('./data/' + args.dataset + '/A_dual.pt').to(device)
#         b_dual = torch.load('./data/' + args.dataset + '/b_dual.pt').to(device)
#
#         D1, D2 = preconditioning(A_dual, args)
#         # scaled_A_dual = D1 @ A_dual @ D2
#         # scaled_b_dual = D1 @ b_dual
#         # D1 D2 are vec(diagonal)
#         scaled_A_dual = D1[:, None] * A_dual * D2
#         scaled_b_dual = D2 * b_dual
#
#         problem = DualLP(scaled_b_dual, scaled_A_dual, args.dual_fx_idx, args.truncate_idx)
#         print(f'A range: {A_dual.min()} - {A_dual.max()}')
#         print(f'scaled_A range: {scaled_A_dual.min()} - {scaled_A_dual.max()}')
#         return problem, D1.detach().cpu(), D2.detach().cpu()
#     else:
#         raise ValueError('Invalid problem')
#
#
# def load_data(args):
#     problem, D1, D2 = load_problem(args)
#
#     input_train = torch.load('./data/' + args.dataset + '/train/input_train.pt')
#     # D1 D2 are vec(diagonal)
#     # input_train = input_train @ D1.t()
#     input_train = input_train * D1
#
#     if args.self_supervised:
#         extension = 'self_'
#         target_train = torch.zeros(input_train.shape[0])
#     else:
#         extension = ''
#         target_train = torch.load('./data/' + args.dataset + '/train/' + extension + 'target_train.pt')
#
#     if args.test_val_train == 'test':
#         input_val = torch.load('./data/' + args.dataset + '/test/input_test.pt')
#         target_val = torch.load('./data/' + args.dataset + '/test/' + extension + 'target_test.pt')
#     elif args.test_val_train == 'val':
#         input_val = torch.load('./data/' + args.dataset + '/val/input_val.pt')
#         target_val = torch.load('./data/' + args.dataset + '/val/' + extension + 'target_val.pt')
#     elif args.test_val_train == 'train':
#         input_val = input_train
#         target_val = target_train
#     else:
#         raise ValueError('Invalid test_val_train')
#     #input_val = input_val @ D1.t()
#     # D1 D2 are vec(diagonal)
#     input_val = input_val * D1
#
#     train_shape_in = input_train.shape
#     train_shape_out = target_train.shape
#     val_shape_in = input_val.shape
#     val_shape_out = target_val.shape
#     print(f'train_shape_in: {train_shape_in}\n'
#           f'train_shape_out: {train_shape_out}\n'
#           f'val_shape_in: {val_shape_in}\n'
#           f'val_shape_out: {val_shape_out}')
#     # mean, std
#     mean = input_train.mean(dim=-1)
#     std = input_train.std(dim=-1)
#
#     train_data = TensorDataset(input_train, target_train)
#     val_data = TensorDataset(input_val, target_val)
#     train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
#     if args.test_val_train == 'test':
#         val = DataLoader(val_data, batch_size=1, shuffle=False)
#     else:
#         val = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
#
#     data_dict = {'train': train, 'val': val,
#                  'mean': mean, 'std': std,
#                  'train_shape_in': train_shape_in, 'train_shape_out': train_shape_out,
#                  'val_shape_in': val_shape_in, 'val_shape_out': val_shape_out}
#     return data_dict, problem


# def load_model(args, problem):
#     input_dim = problem.truncate_idx[1] - problem.truncate_idx[0]
#     output_dim = problem.var_num
#     if 'primal' in problem.name:
#         hidden_dim = args.primal_hidden_dim
#         hidden_num = args.primal_hidden_num
#     elif 'dual' in problem.name:
#         hidden_dim = args.dual_hidden_dim
#         hidden_num = args.dual_hidden_num
#     else:
#         raise ValueError('Invalid problem')
#
#     if args.model == 'primal_nn':
#         Wz_proj, Wb_proj = load_W_proj(args, problem)
#         model = models.OptProjNN(input_dim=input_dim, hidden_dim=hidden_dim,
#                                  hidden_num=hidden_num, output_dim=output_dim,
#                                  truncate_idx=problem.truncate_idx, free_idx=problem.free_idx,
#                                  A=problem.A,
#                                  WzProj=Wz_proj, WbProj=Wb_proj,
#                                  max_iter=args.max_iter, eq_tol=args.eq_tol, ineq_tol= args.ineq_tol,
#                                  proj_method=args.projection, rho=args.rho)
#
#     elif args.model == 'dual_nn':
#         Wz_proj, Wb_proj = load_W_proj(args, problem)
#         model = models.DualOptProjNN(input_dim=input_dim, hidden_dim=hidden_dim,
#                                      hidden_num=hidden_num, output_dim=output_dim,
#                                      truncate_idx=problem.truncate_idx, free_idx=problem.free_idx,
#                                      A=problem.A,
#                                      WzProj=Wz_proj, WbProj=Wb_proj,
#                                      max_iter=args.max_iter, eq_tol=args.eq_tol, ineq_tol= args.ineq_tol,
#                                      projection=args.projection, rho=args.rho,
#                                      b_dual=problem.b)
#
#     elif args.model == 'vanilla_nn':
#         model = models.VanillaNN(input_dim=input_dim, hidden_dim=hidden_dim,
#                                  hidden_num=hidden_num, output_dim=output_dim,
#                                  truncate_idx=problem.truncate_idx)
#
#     else:
#         raise ValueError('Invalid model')
#     model = model.double().to(device)
#     return model


# def get_optimizer(args, model, proj=False):
#     if proj:
#         params = model.projection_layers.parameters()
#         print(f'proj_optimizer for projection_layers')
#         # params = list(model[0].parameters()) + list(model[1].parameters())
#     else:
#         if args.learn2proj:
#             params = model.optimality_layers.parameters()
#             print(f'optimizer for optimality_layers')
#         else:
#             params = model.parameters()
#             print(f'optimizer for all layers')
#
#     if args.optimizer == 'Adam':
#         optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
#     elif args.optimizer == 'SGD':
#         optimizer = optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay)
#     else:
#         raise ValueError('Invalid optimizer')
#     return optimizer


def get_optimizer_new(args, model):
    """
    register_post_accumulate_grad_hook to save gradient memory
    https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html
    """
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


def get_loss(z_star, z1, alpha, targets, inputs, problem, args, loss_type):
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
    elif loss_type == 'obj_alpha':
        return (predicted_obj + problem.obj_example * alpha).mean()
    else:
        raise ValueError('Invalid loss_type')


def process_for_training(inputs, targets, args):
    inputs = inputs.to(device)
    targets = targets.to(device)
    return inputs, targets




