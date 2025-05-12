import torch
import torch.nn as nn
import numpy as np
import models
from problem import *
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import json
import math
import random
import os


def load_problem(args):
    if args.problem == "primal_lp":
        c = torch.load(f'./data/{args.dataset}/new_feasibility/c_backbone.pt').to(args.device)
        nonnegative_mask = torch.load(f'./data/{args.dataset}/new_feasibility/nonnegative_mask.pt')
        problem = PrimalLP(c=c, nonnegative_mask=nonnegative_mask)
    else:
        raise ValueError('Invalid problem')
    return problem


def load_model(args):

    act_cls_mapping = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU, 'elu': nn.ELU,
                       'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'softmax': nn.Softmax}

    mlp = models.MLP(features_dim=args.feature_dim,
                     hidden_dims=args.hidden_dims,
                     out_dim=args.out_dim,
                     act_cls=act_cls_mapping[args.act_cls],
                     batch_norm=args.batch_norm)

    return mlp.to(args.device)


def load_algo(args):
    nonnegative_mask = torch.load(f'./data/{args.dataset}/new_feasibility/nonnegative_mask.pt').to(args.device)

    b_backbone = torch.load(f'./data/{args.dataset}/new_feasibility/b_backbone.pt').to(args.device)
    A_backbone = torch.load(f'./data/{args.dataset}/new_feasibility/A_backbone.pt').to(args.device)

    if args.algo == 'LDRPM':
        ldr_weight = torch.load(f'./data/{args.dataset}/new_feasibility/ldr_weight.pt').to(args.device)
        ldr_bias = torch.load(f'./data/{args.dataset}/new_feasibility/ldr_bias.pt').to(args.device)
        eq_weight, eq_bias_transform = compute_eq_projector(A_backbone)
        algo = models.LDRPM(nonnegative_mask=nonnegative_mask,
                            eq_weight=eq_weight, eq_bias_transform=eq_bias_transform,
                            ldr_weight=ldr_weight, ldr_bias=ldr_bias, ldr_temp=args.ldr_temp)

    elif args.algo == 'POCS':
        eq_weight, eq_bias_transform = compute_eq_projector(A_backbone)
        algo = models.POCS(nonnegative_mask=nonnegative_mask.bool(),
                           eq_weight=eq_weight, eq_bias_transform=eq_bias_transform)

    elif args.algo == 'DC3':
        algo = models.DC3(A=A_backbone, nonnegative_mask=nonnegative_mask,
                          lr=args.dc3_lr, momentum=args.dc3_momentum,
                          changing_feature=args.changing_feature)

    # elif args.algo == 'OPTNET':
    #     algo = models.OPTNET(nonnegative_mask=nonnegative_mask,
    #                          constr_num=args.constr_num, var_num=args.var_num)

    else:
        raise ValueError(f"Invalid algorithm: {args.algo}")

    algo = algo.to(args.device)
    print(f"Loaded algorithm: {args.algo}")
    return models.FeasibilityNet(algo=algo,
                                 eq_tol=args.eq_tol, ineq_tol=args.ineq_tol, max_iters=args.max_iters,
                                 changing_feature=args.changing_feature).to(args.device)


def compute_eq_projector(A):
    A = A.to_sparse() if not A.is_sparse else A
    with torch.no_grad():
        PD = torch.sparse.mm(A, A.t())
        chunk = torch.sparse.mm(A.t(), torch.inverse(PD.to_dense()))
        eq_weight = torch.eye(A.shape[-1]).to(A.device) - torch.sparse.mm(chunk, A)
        eq_bias_transform = chunk
    return eq_weight, eq_bias_transform


def load_instances(args, b_scale, A_scale, b_backbone, A_backbone, train_val_test):
    features = torch.load(f'./data/{args.dataset}/new_feasibility/{train_val_test}/features.pt')
    features = torch.cat([torch.ones((len(features), 1)), features], dim=1)  # add bias term
    targets = torch.load(f'./data/{args.dataset}/new_feasibility/{train_val_test}/targets.pt')

    print(f'Loaded {train_val_test} data: {len(features)} instances')

    dataset = []
    for i in range(len(features)):
        feature = features[i]
        target = targets[i]
        if args.changing_feature == 'b':
            b = b_scale @ feature
            A = A_backbone.clone().detach()
        elif args.changing_feature == 'A':
            b = b_backbone.clone().detach()
            broadcast = torch.cat([f * torch.eye(args.var_num) for f in feature], dim=0)
            A = torch.sparse.mm(A_scale, broadcast)
        else:
            raise ValueError('Invalid changing_feature')

        A_sparse = A.to_sparse() if not A.is_sparse else A
        A_indices = A_sparse.indices()
        A_values = A_sparse.values()
        dataset.append(BasicData(feature=feature[1:], target=target,
                                 b=b, A_indices=A_indices, A_values=A_values,
                                 constr_num=args.constr_num, var_num=args.var_num))

    if train_val_test == 'train':
        bsz = args.batch_size
        shuffle = True
    elif train_val_test == 'val':
        bsz = args.batch_size
        shuffle = False
    elif train_val_test == 'test':
        bsz = 1
        shuffle = False
    else:
        raise ValueError('Invalid train_val_test')
    return DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=0, pin_memory=False, persistent_workers=False)


def load_data(args):
    b_backbone = torch.load(f'./data/{args.dataset}/new_feasibility/b_backbone.pt')
    A_backbone = torch.load(f'./data/{args.dataset}/new_feasibility/A_backbone.pt')

    if args.changing_feature == 'b':
        b_scale = torch.load(f'./data/{args.dataset}/new_feasibility/b_scale.pt')
        A_scale = None
        b_backbone = None
        A_backbone = torch.load(f'./data/{args.dataset}/new_feasibility/A_backbone.pt')
    elif args.changing_feature == 'A':
        b_scale = None
        A_scale = torch.load(f'./data/{args.dataset}/new_feasibility/A_scale.pt')
        b_backbone = torch.load(f'./data/{args.dataset}/new_feasibility/b_backbone.pt')
        A_backbone = None
    else:
        raise ValueError('Invalid changing_feature')

    if args.job in ['training']:
        if args.data_generator:
            # train = InstanceGenerator(args,
            #                           b_scale, A_scale,
            #                           b_backbone, A_backbone)
            train = DataLoader(
                InstanceDataset(args, b_scale, A_scale, b_backbone, A_backbone),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,  # keep RAM low
                pin_memory=False,
                persistent_workers=False,
                exclude_keys=['constr_num', 'var_num']
            )
        else:
            train = load_instances(args, b_scale, A_scale,
                                      b_backbone, A_backbone, 'train')
        val = load_instances(args, b_scale, A_scale,
                                      b_backbone, A_backbone, 'val')
        test = load_instances(args, b_scale, A_scale,
                                      b_backbone, A_backbone, 'test')
        data = {'train': train, 'val': val, 'test': test}
        return data


class BasicData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'A_indices':
            return torch.tensor([[self.constr_num], [self.var_num]])
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ['A_indices']:
            return 1
        if key in ['A_values']:
            return 0
        if key in ['feature', 'b']:
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


class InstanceDataset(torch.utils.data.Dataset):
    def __init__(self, args, b_scale, A_scale, b_backbone, A_backbone):
        self.args          = args
        self.b_scale       = b_scale
        self.A_scale       = A_scale
        self.b_backbone    = b_backbone
        self.A_backbone    = A_backbone
        self.N             = args.bsz_factor * args.batch_size   # total samples/epoch

        self.feature_lb = torch.ones(1 + args.feature_num)
        self.feature_ub = torch.cat([torch.ones(1), -torch.ones(args.feature_num)])

        self._seed = torch.seed()

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        feature = torch.rand_like(self.feature_lb) * (self.feature_ub - self.feature_lb) + self.feature_lb

        if self.args.changing_feature == "b":
            b = self.b_scale @ feature
            A = self.A_backbone
        # else:   # changing A
        #     b = self.b_backbone
        #     broadcast = torch.cat([f * torch.eye(self.args.var_num, device=b.device) for f in feature], 0)
        #     A = torch.sparse.mm(self.A_scale, broadcast)

        A = A.to_sparse() if not A.is_sparse else A

        return BasicData(            # exactly what your model expected before
            feature    = feature[1:],
            target     = torch.zeros(1),
            b          = b,
            A_indices  = A.indices(),
            A_values   = A.values(),
            constr_num = self.args.constr_num,
            var_num    = self.args.var_num,
        )

    def refresh(self):
        self._seed = torch.seed()


# class InstanceDataset(torch.utils.data.IterableDataset):
#     def __init__(self, args,
#                  b_scale, A_scale,
#                  b_backbone, A_backbone):
#         super().__init__()
#         self.bsz_factor = args.bsz_factor
#         self.batch_size = args.batch_size
#         self.feature_lb = torch.ones(1+args.feature_num)
#         self.feature_ub = torch.cat([torch.ones(1), - torch.ones(args.feature_num)])
#         self.b_scale = b_scale
#         self.A_scale = A_scale
#         self.b_backbone = b_backbone
#         self.A_backbone = A_backbone
#
#         self.constr_num = args.constr_num
#         self.var_num = args.var_num
#         self.changing_feature = args.changing_feature
#
#     def __iter__(self):
#         for _ in range(self.bsz_factor * self.batch_size):
#             feature = torch.rand_like(self.feature_lb) * (self.feature_ub - self.feature_lb) + self.feature_lb
#
#             if self.args.changing_feature == "b":
#                 yield self._build(feature, self.b_scale @ feature, self.A_backbone)
#             # else:                                        # changing A
#             #     b = self.b_backbone
#             #     broadcast = torch.cat([f * torch.eye(self.args.var_num) for f in feature], dim=0)
#             #     A   = torch.sparse.mm(self.A_scale, broadcast)
#             #     yield self._build(feature, b, A)
#
#     def _build(self, feature, b, A):
#         A = A.to_sparse() if not A.is_sparse else A
#         return BasicData(
#             feature=feature[1:],
#             target=torch.zeros(1),
#             b=b,
#             A_indices=A.indices(),
#             A_values=A.values(),
#             constr_num=self.args.constr_num,
#             var_num=self.args.var_num,
#         )


# class InstanceGenerator:
#     def __init__(self, args,
#                  b_scale, A_scale,
#                  b_backbone, A_backbone):
#         self.bsz_factor = args.bsz_factor
#         self.batch_size = args.batch_size
#         self.feature_lb = torch.ones(1+args.feature_num)
#         self.feature_ub = torch.cat([torch.ones(1), - torch.ones(args.feature_num)])
#         self.b_scale = b_scale
#         self.A_scale = A_scale
#         self.b_backbone = b_backbone
#         self.A_backbone = A_backbone
#
#         self.constr_num = args.constr_num
#         self.var_num = args.var_num
#         self.changing_feature = args.changing_feature
#
#         self.data_structure = BasicData()
#
#         self.data = None
#         self.reset_data()
#
#     def _generate_data(self):
#         num_instances = self.bsz_factor * self.batch_size
#         train_dataset = []
#
#         for _ in range(num_instances):
#             feature = torch.rand_like(self.feature_lb) * (self.feature_ub - self.feature_lb) + self.feature_lb
#
#             if self.changing_feature == 'b':
#                 b = self.b_scale @ feature
#                 A = self.A_backbone.clone().detach()
#             elif self.changing_feature == 'A':
#                 b = self.b_backbone.clone().detach()
#                 broadcast = torch.cat([f * torch.eye(self.var_num) for f in feature], dim=0)
#                 A = torch.sparse.mm(self.A_scale, broadcast)
#             else:
#                 raise ValueError('Invalid changing_feature')
#
#             target = torch.zeros(1)
#
#             A_sparse = A.to_sparse() if not A.is_sparse else A
#             A_indices = A_sparse.indices()
#             A_values = A_sparse.values()
#             train_dataset.append(BasicData(feature=feature[1:], target=target,
#                                            b=b, A_indices=A_indices, A_values=A_values,
#                                            constr_num=self.constr_num,
#                                            var_num=self.var_num))
#
#         return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
#                           exclude_keys=['constr_num', 'var_num'], num_workers=0, pin_memory=False, persistent_workers=False)
#
#     def reset_data(self):
#         self.data = self._generate_data()


def get_optimizer(args, model):
    params = model.parameters()
    if args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError('Invalid optimizer')
    return optimizer


# bsz = 32
# constr_num = 20
# var_num = 50
# A_backbone = torch.rand((constr_num, var_num))
# b_backbone = torch.rand((constr_num,))
# A_feature_mask = torch.randint(0, 2, (constr_num,var_num)).bool()
# b_feature_mask = torch.randint(0, 2, (constr_num,)).bool()
# A_feature_num = A_feature_mask.sum().item()
# b_feature_num = b_feature_mask.sum().item()
# b_feature_lb = torch.ones(b_feature_num) * 0.1
# b_feature_ub = torch.ones(b_feature_num) * 2.5
# A_feature_lb = torch.ones(A_feature_num) * 0.3
# A_feature_ub = torch.ones(A_feature_num) * 3.5
# class args:
#     bsz_factor = 3
#     batch_size = bsz
# args = args()
#
# # InstanceGenerator()
# dataloader = InstanceGenerator(args, (b_feature_lb, b_feature_ub),
#                                (A_feature_lb, A_feature_ub),
#                                b_backbone, A_backbone,
#                                b_feature_mask, A_feature_mask)
# data = dataloader.data
# data = iter(data)
# batch = next(data)
# A_sparse = torch.sparse_coo_tensor(batch.A_indices, batch.A_values, (bsz * constr_num, bsz * var_num))
