import torch
import pandas as pd
import numpy
import os
import pickle


# def get_self_supervised_data(args):
#     if args.dataset == 'DCOPF':
#         csv_path = 'data/' + args.dataset + '/b.csv'
#         df = pd.read_csv(csv_path, header=None)
#         b_ref = torch.tensor(df.values).squeeze()
#
#         data_points = []
#         for _ in range(args.num_points):
#             # Generate random numbers in the range
#             random_numbers = args.random_range[0] + torch.rand(b_ref.size()) * (args.random_range[1] - args.random_range[0])
#             # Multiply b_ref with random numbers
#             b = b_ref * random_numbers
#             data_points.append(b)
#         # Stack all data points into a matrix
#         dataset = torch.stack(data_points)
#         # Save the dataset
#         torch.save(dataset, 'data/' + args.dataset + '/' + args.train_val_test + '/input_' + args.train_val_test + '.pt')
#
#     elif args.dataset == 'DCOPF_debug':
#         # b: 10000 x 191 (will be input);
#         # x: 10000 x 288 (will be supervised learning target);
#         # cost: 10000 x 1 (will be self_supervised learning target);
#         csv_dir = 'data/' + args.dataset
#         b = pd.read_csv(csv_dir + '/b_true.csv', header=None)
#         x = pd.read_csv(csv_dir + '/x_true.csv', header=None)
#         cost = pd.read_csv(csv_dir + '/cost_true.csv', header=None)
#         print(b.shape, x.shape, cost.shape)
#         b = torch.tensor(b.values).float()
#         x = torch.tensor(x.values).float()
#         cost = torch.tensor(cost.values).float().squeeze()
#         # split into 80% train and 20% val
#         train_size = int(0.8 * b.size(0))
#         val_size = b.size(0) - train_size
#         print(train_size, val_size)
#         # save the data
#         torch.save(b[:train_size], csv_dir + '/train/input_train.pt')
#         torch.save(x[:train_size], csv_dir + '/train/target_train.pt')
#         torch.save(cost[:train_size], csv_dir + '/train/self_target_train.pt')
#         torch.save(b[train_size:], csv_dir + '/val/input_val.pt')
#         torch.save(x[train_size:], csv_dir + '/val/target_val.pt')
#         torch.save(cost[train_size:], csv_dir + '/val/self_target_val.pt')
#
#
# def get_blank_obj(args):
#     if args.dataset == 'DCOPF':
#         vector = torch.zeros(args.num_points)
#         torch.save(vector, 'data/' + args.dataset + '/' + args.train_val_test + '/target_' + args.train_val_test + '.pt')
#
#
# def get_ref_A(args):
#     if args.dataset == 'DCOPF':
#         # Generate the reduced A matrix
#         csv_path = 'data/' + args.dataset + '/A.csv'
#         df = pd.read_csv(csv_path, header=None)
#         A_ref = torch.tensor(df.values)
#         torch.save(A_ref, 'data/' + args.dataset + '/Aref.pt')
#
#
# def get_c(args):
#     if args.dataset == 'DCOPF':
#         # Generate the c vector
#         csv_path = 'data/' + args.dataset + '/c.csv'
#         df = pd.read_csv(csv_path, header=None)
#         c = torch.tensor(df.values).squeeze()
#         torch.save(c, 'data/' + args.dataset + '/c.pt')


# dataset = 'DCOPF'
#
# csv_path = 'data/' + dataset + '/A_primal.csv'
# df = pd.read_csv(csv_path, header=None)
# A_primal = torch.tensor(df.values)
#
# csv_path = 'data/' + dataset + '/c_primal.csv'
# df = pd.read_csv(csv_path, header=None)
# c_primal = torch.tensor(df.values).squeeze()
#
# csv_path = 'data/' + dataset + '/A_dual.csv'
# df = pd.read_csv(csv_path, header=None)
# A_dual = torch.tensor(df.values)
#
# csv_path = 'data/' + dataset + '/b_dual.csv'
# df = pd.read_csv(csv_path, header=None)
# b_dual = torch.tensor(df.values).squeeze()
#
#
# torch.save(A_primal, 'data/' + dataset + '/A_primal.pt')
# torch.save(c_primal, 'data/' + dataset + '/c_primal.pt')
# torch.save(A_dual, 'data/' + dataset + '/A_dual.pt')
# torch.save(b_dual, 'data/' + dataset + '/b_dual.pt')
#
#
# csv_path = 'data/' + dataset + '/b_primal.csv'
# df = pd.read_csv(csv_path, header=None)
# b_primal = torch.tensor(df.values)
#
# csv_path = 'data/' + dataset + '/c_dual.csv'
# df = pd.read_csv(csv_path, header=None)
# c_dual = torch.tensor(df.values).squeeze()
#
# csv_path = 'data/' + dataset + '/cost_true.csv'
# df = pd.read_csv(csv_path, header=None)
# cost_true = torch.tensor(df.values).squeeze()
#
# train_size = int(0.8 * b_primal.size(0))
# val_size = b_primal.size(0) - train_size
# torch.save(b_primal[:train_size], 'data/' + dataset + '/train/input_train.pt')
# torch.save(c_dual[:train_size], 'data/' + dataset + '/train/dual_input_train.pt')


# torch.save(cost_true[:train_size], 'data/' + dataset + '/train/self_target_train.pt')
# torch.save(b_primal[train_size:], 'data/' + dataset + '/val/input_val.pt')
# torch.save(c_dual[train_size:], 'data/' + dataset + '/val/dual_input_val.pt')
# torch.save(cost_true[train_size:], 'data/' + dataset + '/val/self_target_val.pt')

# torch.save(primal_true[:train_size], 'data/' + dataset + '/train/target_train.pt')
# torch.save(primal_true[train_size:], 'data/' + dataset + '/val/target_val.pt')


def get_DCOPF_data_tensor():
    dataset = 'DCOPF_'
    csv_path = 'data/' + dataset + '/'
    A_primal = torch.tensor(pd.read_csv(csv_path + 'A_primal.csv', header=None).values)
    c_primal = torch.tensor(pd.read_csv(csv_path + 'c_primal.csv', header=None).values).squeeze()
    A_dual = torch.tensor(pd.read_csv(csv_path + 'A_dual.csv', header=None).values)
    b_dual = torch.tensor(pd.read_csv(csv_path + 'b_dual.csv', header=None).values).squeeze()
    b_primal = torch.tensor(pd.read_csv(csv_path + 'b_primal.csv', header=None).values)
    c_dual = torch.tensor(pd.read_csv(csv_path + 'c_dual.csv', header=None).values).squeeze()
    cost_true = torch.tensor(pd.read_csv(csv_path + 'cost_true.csv', header=None).values).squeeze()
    #primal_true = torch.tensor(pd.read_csv(csv_path + 'primal_true.csv', header=None).values)
    #dual_true = torch.tensor(pd.read_csv(csv_path + 'dual_true.csv', header=None).values)





    torch.save(A_primal.detach().cpu(), csv_path + 'A_primal.pt')
    torch.save(c_primal.detach().cpu(), csv_path + 'c_primal.pt')
    torch.save(A_dual.detach().cpu(), csv_path + 'A_dual.pt')
    torch.save(b_dual.detach().cpu(), csv_path + 'b_dual.pt')

    train_size = int(0.8 * b_primal.size(0))
    val_size = (b_primal.size(0) - train_size) // 2
    test_size = b_primal.size(0) - train_size - val_size

    input_train, input_val, input_test = torch.split(b_primal, [train_size, val_size, test_size], dim=0)
    #dual_input_train, dual_input_val, dual_input_test = torch.split(c_dual, [train_size, val_size, test_size], dim=0)
    self_target_train, self_target_val, self_target_test = torch.split(cost_true, [train_size, val_size, test_size], dim=0)
    #target_train, target_val, target_test = torch.split(primal_true, [train_size, val_size, test_size], dim=0)
    #dual_target_train, dual_target_val, dual_target_test = torch.split(dual_true, [train_size, val_size, test_size], dim=0)

    torch.save(input_train.detach().cpu(), csv_path + 'train/input_train.pt')
    torch.save(input_val.detach().cpu(), csv_path + 'val/input_val.pt')
    torch.save(input_test.detach().cpu(), csv_path + 'test/input_test.pt')

    # torch.save(dual_input_train.detach().cpu(), csv_path + 'train/dual_input_train.pt')
    # torch.save(dual_input_val.detach().cpu(), csv_path + 'val/dual_input_val.pt')
    # torch.save(dual_input_test.detach().cpu(), csv_path + 'test/dual_input_test.pt')

    torch.save(self_target_train.detach().cpu(), csv_path + 'train/self_target_train.pt')
    torch.save(self_target_val.detach().cpu(), csv_path + 'val/self_target_val.pt')
    torch.save(self_target_test.detach().cpu(), csv_path + 'test/self_target_test.pt')

    # torch.save(target_train.detach().cpu(), csv_path + 'train/target_train.pt')
    # torch.save(target_val.detach().cpu(), csv_path + 'val/target_val.pt')
    # torch.save(target_test.detach().cpu(), csv_path + 'test/target_test.pt')

    # torch.save(dual_target_train.detach().cpu(), csv_path + 'train/dual_target_train.pt')
    # torch.save(dual_target_val.detach().cpu(), csv_path + 'val/dual_target_val.pt')
    # torch.save(dual_target_test.detach().cpu(), csv_path + 'test/dual_target_test.pt')

def get_case39_data_tensor():
    dataset = 'case39'
    csv_path = 'data/' + dataset + '/'
    A_primal = torch.tensor(pd.read_csv(csv_path + f'A_{dataset}.csv', header=None).values)
    c_primal = torch.tensor(pd.read_csv(csv_path + f'c_{dataset}.csv', header=None).values).squeeze()
    b_primal = torch.tensor(pd.read_csv(csv_path + f'btest_{dataset}.csv', header=None).values).t()
    Q_LDR = torch.tensor(pd.read_csv(csv_path + f'Q_{dataset}.csv', header=None).values)
    z0_LDR = torch.tensor(pd.read_csv(csv_path + f'z0_{dataset}.csv', header=None).values).squeeze()
    cost_true = torch.tensor(pd.read_csv(csv_path + f'cost_true_{dataset}.csv', header=None).values).squeeze()

    torch.save(A_primal.detach().cpu(), csv_path + 'A_primal_dense.pt')
    torch.save(c_primal.detach().cpu(), csv_path + 'c_primal.pt')
    torch.save(Q_LDR.detach().cpu(), csv_path + 'Q_LDR.pt')
    torch.save(z0_LDR.detach().cpu(), csv_path + 'z0_LDR.pt')

    Q_opt_LDR = torch.tensor(pd.read_csv(csv_path + f'Q_opt_{dataset}.csv', header=None).values)
    z0_opt_LDR = torch.tensor(pd.read_csv(csv_path + f'z0_opt_{dataset}.csv', header=None).values).squeeze()
    torch.save(Q_opt_LDR.detach().cpu(), csv_path + 'Q_opt_LDR.pt')
    torch.save(z0_opt_LDR.detach().cpu(), csv_path + 'z0_opt_LDR.pt')

    train_size = int(0.8 * b_primal.size(0))
    val_size = (b_primal.size(0) - train_size) // 2
    test_size = b_primal.size(0) - train_size - val_size

    input_train, input_val, input_test = torch.split(b_primal, [train_size, val_size, test_size], dim=0)
    self_target_train, self_target_val, self_target_test = torch.split(cost_true, [train_size, val_size, test_size], dim=0)


    torch.save(input_train.detach().cpu(), csv_path + 'train/input_train.pt')
    torch.save(input_val.detach().cpu(), csv_path + 'val/input_val.pt')
    torch.save(input_test.detach().cpu(), csv_path + 'test/input_test.pt')

    torch.save(self_target_train.detach().cpu(), csv_path + 'train/self_target_train.pt')
    torch.save(self_target_val.detach().cpu(), csv_path + 'val/self_target_val.pt')
    torch.save(self_target_test.detach().cpu(), csv_path + 'test/self_target_test.pt')

def get_Smallest_data_tensor():
    dataset = 'Smallest'
    csv_path = 'data/' + dataset + '/'
    A_primal = torch.tensor(pd.read_csv(csv_path + 'A_case14.csv', header=None).values)
    c_primal = torch.tensor(pd.read_csv(csv_path + 'c_case14.csv', header=None).values).squeeze()
    b_primal = torch.tensor(pd.read_csv(csv_path + 'b_case14.csv', header=None).values).view(1, -1)
    Q_LDR = torch.tensor(pd.read_csv(csv_path + 'Q.csv', header=None).values)
    z0_LDR = torch.tensor(pd.read_csv(csv_path + 'z0.csv', header=None).values).squeeze()

    torch.save(A_primal.detach().cpu(), csv_path + 'A_primal.pt')
    torch.save(c_primal.detach().cpu(), csv_path + 'c_primal.pt')
    torch.save(b_primal.detach().cpu(), csv_path + 'b_primal.pt')  # note that we only have one b
    torch.save(Q_LDR.detach().cpu(), csv_path + 'Q_LDR.pt')
    torch.save(z0_LDR.detach().cpu(), csv_path + 'z0_LDR.pt')

    train_size = int(0.8 * b_primal.size(0))
    val_size = (b_primal.size(0) - train_size) // 2
    test_size = b_primal.size(0) - train_size - val_size

    # input_train, input_val, input_test = torch.split(b_primal, [train_size, val_size, test_size], dim=0)
    # self_target_train, self_target_val, self_target_test = torch.split(cost_true, [train_size, val_size, test_size], dim=0)

    input_train, input_val, input_test = b_primal, b_primal, b_primal
    cost_true = torch.zeros((1,))
    self_target_train, self_target_val, self_target_test = cost_true, cost_true, cost_true

    torch.save(input_train.detach().cpu(), csv_path + 'train/input_train.pt')
    torch.save(input_val.detach().cpu(), csv_path + 'val/input_val.pt')
    torch.save(input_test.detach().cpu(), csv_path + 'test/input_test.pt')

    torch.save(self_target_train.detach().cpu(), csv_path + 'train/self_target_train.pt')
    torch.save(self_target_val.detach().cpu(), csv_path + 'val/self_target_val.pt')
    torch.save(self_target_test.detach().cpu(), csv_path + 'test/self_target_test.pt')


def get_DC3_data_csv(dataset):
    # 'simpleDC3' or 'noncvxQP'
    csv_path = 'data/' + dataset + '/'
    Q = torch.tensor(pd.read_csv(csv_path + 'Q.csv', index_col=0).values)
    p = torch.tensor(pd.read_csv(csv_path + 'p.csv', index_col=0).values).squeeze()
    A = torch.tensor(pd.read_csv(csv_path + 'A.csv', index_col=0).values)
    G = torch.tensor(pd.read_csv(csv_path + 'G.csv', index_col=0).values)
    h = torch.tensor(pd.read_csv(csv_path + 'h.csv', index_col=0).values).squeeze()
    X = torch.tensor(pd.read_csv(csv_path + 'X.csv', index_col=0).values)
    Y = torch.tensor(pd.read_csv(csv_path + 'Y.csv', index_col=0).values)
    # figure out slacks s first
    S = h - torch.matmul(Y, G.T)  # 10000 x 50
    new_A = torch.cat((torch.cat((A, torch.zeros(50, 50)), dim=1),
                       torch.cat((G, torch.eye(50)), dim=1)
                       ), dim=0)
    new_X = torch.cat((X, h.unsqueeze(0).expand(X.size(0), -1)), dim=1)
    new_Y = torch.cat((Y, S), dim=1)
    new_c = torch.cat((p, torch.zeros(50)), dim=0)
    new_Q = torch.cat((torch.cat((Q, torch.zeros(100, 50)), dim=1), torch.zeros(50, 150)), dim=0)

    if dataset == 'simpleDC3':
        cost_true = torch.sum(1 / 2 * torch.mm(new_Y, new_Q) * new_Y, dim=1) + torch.matmul(new_Y, new_c)
    elif dataset == 'noncvxQP':
        cost_true = torch.sum(1 / 2 * torch.mm(new_Y, new_Q) * new_Y, dim=1) + torch.matmul(torch.sin(new_Y), new_c)

    pd.DataFrame(new_A.detach().cpu().numpy()).to_csv(csv_path + 'new_A.csv', index=False, header=False)
    pd.DataFrame(new_X.detach().cpu().numpy()).to_csv(csv_path + 'new_X.csv', index=False, header=False)
    pd.DataFrame(new_Y.detach().cpu().numpy()).to_csv(csv_path + 'new_Y.csv', index=False, header=False)
    pd.DataFrame(new_c.detach().cpu().numpy()).to_csv(csv_path + 'new_c.csv', index=False, header=False)
    pd.DataFrame(new_Q.detach().cpu().numpy()).to_csv(csv_path + 'new_Q.csv', index=False, header=False)
    pd.DataFrame(cost_true.detach().cpu().numpy()).to_csv(csv_path + 'cost_true.csv', index=False, header=False)


def get_DC3_data_tensor(dataset):
    csv_path = 'data/' + dataset + '/'
    new_A = torch.tensor(pd.read_csv(csv_path + 'new_A.csv', header=None).values)
    new_X = torch.tensor(pd.read_csv(csv_path + 'new_X.csv', header=None).values)
    new_Y = torch.tensor(pd.read_csv(csv_path + 'new_Y.csv', header=None).values)
    new_c = torch.tensor(pd.read_csv(csv_path + 'new_c.csv', header=None).values).squeeze()
    new_Q = torch.tensor(pd.read_csv(csv_path + 'new_Q.csv', header=None).values)
    cost_true = torch.tensor(pd.read_csv(csv_path + 'cost_true.csv', header=None).values).squeeze()

    torch.save(new_A.detach().cpu(), csv_path + 'new_A.pt')
    torch.save(new_c.detach().cpu(), csv_path + 'new_c.pt')
    torch.save(new_Q.detach().cpu(), csv_path + 'new_Q.pt')

    train_size = int(0.8 * new_X.size(0))
    val_size = (new_X.size(0) - train_size)//2
    test_size = new_X.size(0) - train_size - val_size

    input_train, input_val, input_test = torch.split(new_X, [train_size, val_size, test_size], dim=0)
    self_target_train, self_target_val, self_target_test = torch.split(cost_true, [train_size, val_size, test_size], dim=0)
    target_train, target_val, target_test = torch.split(new_Y, [train_size, val_size, test_size], dim=0)

    torch.save(input_train.detach().cpu(), 'data/' + dataset + '/train/input_train.pt')
    torch.save(input_val.detach().cpu(), 'data/' + dataset + '/val/input_val.pt')
    torch.save(input_test.detach().cpu(), 'data/' + dataset + '/test/input_test.pt')

    torch.save(self_target_train.detach().cpu(), 'data/' + dataset + '/train/self_target_train.pt')
    torch.save(self_target_val.detach().cpu(), 'data/' + dataset + '/val/self_target_val.pt')
    torch.save(self_target_test.detach().cpu(), 'data/' + dataset + '/test/self_target_test.pt')

    torch.save(target_train.detach().cpu(), 'data/' + dataset + '/train/target_train.pt')
    torch.save(target_val.detach().cpu(), 'data/' + dataset + '/val/target_val.pt')
    torch.save(target_test.detach().cpu(), 'data/' + dataset + '/test/target_test.pt')






