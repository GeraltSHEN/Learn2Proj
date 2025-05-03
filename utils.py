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
        c = torch.load(f'./data/{args.dataset}/c_backbone.pt')
        nonnegative_mask = torch.load(f'./data/{args.dataset}/nonnegative_mask.pt')
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

    return mlp


def load_algo(args):
    nonnegative_mask = torch.load(f'./data/{args.dataset}/nonnegative_mask.pt')
    A = torch.load(f'./data/{args.dataset}/A_sparse.pt')

    if args.algo == 'LDRPM':
        ldr_weight = torch.load(f'./data/{args.dataset}/ldr_weight.pt')
        ldr_bias = torch.load(f'./data/{args.dataset}/ldr_bias.pt')
        eq_weight, eq_bias_transform = compute_eq_projector(A)
        algo = models.LDRPM(nonnegative_mask=nonnegative_mask,
                            eq_weight=eq_weight, eq_bias_transform=eq_bias_transform,
                            ldr_weight=ldr_weight, ldr_bias=ldr_bias)

    elif args.algo == 'POCS':
        eq_weight, eq_bias_transform = compute_eq_projector(A)
        algo = models.POCS(nonnegative_mask=nonnegative_mask,
                           eq_weight=eq_weight, eq_bias_transform=eq_bias_transform)

    else:
        raise ValueError(f"Invalid algorithm: {args.algo}")

    return models.FeasibilityNet(algo=algo, eq_tol=args.eq_tol, ineq_tol=args.ineq_tol, max_iters=args.max_iters)


def compute_eq_projector(A):
    assert A.is_sparse
    with torch.no_grad():
        PD = torch.sparse.mm(A, A.t())
        chunk = torch.sparse.mm(A.t(), torch.inverse(PD.to_dense()))
        eq_weight = torch.eye(A.shape[-1]) - torch.sparse.mm(chunk, A)
        eq_bias_transform = chunk
    return eq_weight, eq_bias_transform


def load_instances(args, train_val_test):
    As_indices = torch.load(f'./data/{args.dataset}/As_indices.pt')
    As_values = torch.load(f'./data/{args.dataset}/As_values.pt')
    bs = torch.load(f'./data/{args.dataset}/bs.pt')

    features = torch.load(f'./data/{args.dataset}/{train_val_test}/features.pt')
    targets = torch.load(f'./data/{args.dataset}/{train_val_test}/targets.pt')

    dataset = []
    for i in range(len(features)):
        dataset.append(BasicData(feature=features[i], target=targets[i],
                                 b=bs[i], A_indices=As_indices[i], A_values=As_values[i],
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
    return DataLoader(dataset, batch_size=bsz, shuffle=shuffle)


def load_data(args):
    if args.job in ['training']:
        if args.data_generator:
            b_feature_lb = torch.load(f'./data/{args.dataset}/b_feature_lb.pt')
            b_feature_ub = torch.load(f'./data/{args.dataset}/b_feature_ub.pt')
            A_feature_lb = torch.load(f'./data/{args.dataset}/A_feature_lb.pt')
            A_feature_ub = torch.load(f'./data/{args.dataset}/A_feature_ub.pt')
            b_backbone = torch.load(f'./data/{args.dataset}/b_backbone.pt')
            A_backbone = torch.load(f'./data/{args.dataset}/A_backbone.pt')
            b_feature_mask = torch.load(f'./data/{args.dataset}/b_feature_mask.pt')
            A_feature_mask = torch.load(f'./data/{args.dataset}/A_feature_mask.pt')

            train = InstanceGenerator(args,
                                      (b_feature_lb, b_feature_ub), (A_feature_lb, A_feature_ub),
                                      b_backbone, A_backbone,
                                      b_feature_mask, A_feature_mask)
        else:
            train = load_instances(args, 'train')
        val = load_instances(args, 'val')
        test = load_instances(args, 'test')
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


class InstanceGenerator:
    def __init__(self, args,
                 b_feature_range, A_feature_range,
                 b_backbone, A_backbone,
                 b_feature_mask, A_feature_mask):
        self.bsz_factor = args.bsz_factor
        self.batch_size = args.batch_size
        self.b_feature_lb, self.b_feature_ub = b_feature_range
        self.A_feature_lb, self.A_feature_ub = A_feature_range
        self.b_feature_mask = b_feature_mask
        self.A_feature_mask = A_feature_mask
        self.b_backbone = b_backbone
        self.A_backbone = A_backbone

        self.data_structure = BasicData()

        self.data = None
        self.reset_data()

    def _generate_data(self):
        num_instances = self.bsz_factor * self.batch_size
        train_dataset = []

        for _ in range(num_instances):
            b_feature = torch.rand_like(self.b_feature_lb) * (self.b_feature_ub - self.b_feature_lb) + self.b_feature_lb
            A_feature = torch.rand_like(self.A_feature_lb) * (self.A_feature_ub - self.A_feature_lb) + self.A_feature_lb

            b = self.b_backbone.clone().detach()
            A = self.A_backbone.clone().detach()

            feature = torch.cat([b_feature, A_feature], dim=0)
            target = torch.zeros(1)
            b[self.b_feature_mask] = b_feature
            A[self.A_feature_mask] = A_feature

            A_sparse = A.to_sparse()
            A_indices = A_sparse.indices()
            A_values = A_sparse.values()
            train_dataset.append(BasicData(feature=feature, target=target,
                                           b=b, A_indices=A_indices, A_values=A_values,
                                           constr_num=self.A_backbone.shape[0],
                                           var_num=self.A_backbone.shape[1]))

        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                          exclude_keys=['constr_num', 'var_num'])

    def reset_data(self):
        self.data = self._generate_data()


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
