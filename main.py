from train import run_training
from utils import load_data_new, load_problem_new
from pretrain import rho_search, baseline_pocs, baseline_nullspace, data_sanity_check
import argparse
import os
import torch
import default_args

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="DCOPF, DCOPF_large, simpleDC3, noncvxQP")
    parser.add_argument("--problem", help="primal_lp, dual_lp")
    parser.add_argument("--model", help="primal_nn, dual_nn, vanilla_nn")
    parser.add_argument("--model_id", type=str)

    # dataset related parameters
    parser.add_argument("--truncate_idx", type=tuple)  # default="1,501", "2,40"
    parser.add_argument("--primal_const_num", type=int)  # default=2143, 152
    parser.add_argument("--primal_var_num", type=int)  # default=2313, 161
    parser.add_argument("--primal_fx_idx", type=tuple)  # default=671, 49
    parser.add_argument("--dual_const_num", type=int)  # default=671, 49
    parser.add_argument("--dual_var_num", type=int)  # default=2143, 152
    parser.add_argument("--dual_fx_idx", type=tuple)  # default=501, 40

    # hidden layers related parameters
    parser.add_argument("--primal_hidden_dim", type=int)
    parser.add_argument("--primal_hidden_num", type=int)
    parser.add_argument("--dual_hidden_dim", type=int)
    parser.add_argument("--dual_hidden_num", type=int)
    parser.add_argument("--proj_hidden_dim", type=int)
    parser.add_argument("--proj_hidden_num", type=int)

    # ALM and penalty related parameters
    parser.add_argument("--penalty_g", type=float,
                        help="penalty for slacks < 0, i.e., s >= 0")
    parser.add_argument("--penalty_h", type=float,
                        help="penalty for equality violation Ax - b")

    # training related parameters
    parser.add_argument("--loss_type", type=str)
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--lr", help="learning rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--data_generator", type=bool)
    parser.add_argument("--self_supervised", type=bool)

    # evaluation related parameters
    parser.add_argument("--test_val_train", default="val", type=str)
    parser.add_argument("--job", default="training", type=str)

    # project related parameters
    parser.add_argument("--max_iter", type=int)
    parser.add_argument("--f_tol", type=float)
    parser.add_argument("--eq_tol", type=float)  # not used, calculated using f_tol
    parser.add_argument("--ineq_tol", type=float)  # not used, calculated using f_tol
    parser.add_argument("--projection", type=str)
    parser.add_argument("--rho", type=float)
    parser.add_argument("--learn2proj", type=bool)
    parser.add_argument("--proj_epochs", type=int)
    parser.add_argument("--precondition", type=str)
    parser.add_argument("--periodic", type=bool)

    # save related parameters
    parser.add_argument("--saveAllStats", default=True, type=bool)
    parser.add_argument("--resultSaveFreq", default=50, type=int)

    parser.add_argument("--float64", default=False, type=bool)

    return parser.parse_args()


def complete_args(args):
    defaults = default_args.get_default_args(args.dataset)
    for key in defaults.keys():
        if eval('args.' + key) is None:
            exec('args.' + key + ' = defaults[key]')

    if args.model == 'dual_learn2proj':
        args.learn2proj = True
    else:
        args.learn2proj = False

    args.primal_input_dim = args.truncate_idx[1] - args.truncate_idx[0]
    args.primal_x_dim = args.primal_var_num
    args.dual_input_dim = args.truncate_idx[1] - args.truncate_idx[0]
    args.dual_y_dim = args.dual_var_num

    if args.epochs < args.resultSaveFreq:
        args.resultSaveFreq = args.epochs

    # # def estimate_hidden_dim_and_num(input_dim):
    # #     import math
    # #     n0 = input_dim
    # #     # let hidden_dim be about xxx * n0, round up to the nearest multiple of 8
    # #     n = math.ceil(1.1*n0 / 8) * 8
    # #     # (n/n0) ^ ( (L−1)n0 ) * n^n0 ---> 2^n0
    # #     # ( (L−1)n0 ) * log2(n/n0) + n0 * log2(n) ---> n0
    # #     # find hidden_num, L
    # #     L = 1
    # #     while True:
    # #         if (L-1)*n0 * math.log2(n/n0) + n0 * math.log2(n) >= n0:
    # #             break
    # #         L += 1
    # #     return n, L
    print(args)


def main(args):
    if not os.path.exists('./models'):
        os.makedirs('./models')
    if not os.path.exists('./data/prediction'):
        os.makedirs('./data/prediction')
    if not os.path.exists('./data/results_summary'):
        os.makedirs('./data/results_summary')
    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    if args.float64:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    if args.job == 'training':
        problem = load_problem_new(args)
        data = load_data_new(args, problem)
        # run training
        run_training(args, data, problem)
    elif args.job == 'rho_search':
        rho_search(args)
    elif args.job == 'baseline_pocs':
        baseline_pocs(args)
    elif args.job == 'baseline_nullspace':
        baseline_nullspace(args)
    elif args.job == 'data_sanity_check':
        print('data_sanity_check disabled')
        data_sanity_check(args)
    elif args.job == 'make_datasets':
        pass
    else:
        pass
        # data = load_data(args)
        # # evaluate model
        # evaluate_model(data, args)


if __name__ == '__main__':
    args = add_arguments()
    complete_args(args)
    main(args)