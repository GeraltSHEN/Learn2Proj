from utils import *
import torch
import torch.nn as nn
import csv
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import random
import time
import os
import pickle


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)


def run_pretraining(args, data, problem):
    model = load_model(args, problem)
    print(f'----- {args.model_id} Learn to Project -----')
    print('#params:', sum(p.numel() for p in model.parameters()))
    optimizer = get_optimizer(args, model)
    loss_fn = nn.MSELoss()

    if args.data_generator:
        print('self-generated data training code has not been implemented yet')
    else:
        print('training dataset is given already')

    start_time = time.time()
    ##############################################################################################################
    Learning(args, data, model, optimizer, loss_fn)
    ##############################################################################################################
    end_time = time.time()
    training_time = end_time - start_time
    print(f'----time required for {args.epochs} epochs training: {round(training_time)}s----')
    print(f'----time required for {args.epochs} epochs training: {round(training_time / 60)}min----')
    print(f'----time required for {args.epochs} epochs training: {round(training_time / 3600)}hr----')

    # check the model on the test set
    print(f'switching {args.test_val_train} dataset ')
    args.test_val_train = 'test'
    print(f'to {args.test_val_train} dataset')
    data = load_data(args)
    scores = evaluate_model(args, data, problem)


def Learning(args, data, model, optimizer, loss_fn):
    best = np.inf
    for epoch in range(args.epochs):
        running_loss = 0
        running_val_loss = 0
        model.train()
        for (inputs, targets) in data['train']:
            inputs, targets = process_for_training(inputs, targets, args)
            optimizer.zero_grad()
            z_projected = model(inputs)
            loss = loss_fn(z_projected, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().cpu()
        loss = running_loss / len(data['train'])

        model.eval()
        for (inputs, targets) in data['val']:
            inputs, targets = process_for_training(inputs, targets, args)
            z_projected = model(inputs)
            val_loss = loss_fn(z_projected, targets)
            running_val_loss += val_loss.detach().cpu()
        val_loss = running_val_loss / len(data['val'])

        print('Epoch {}, Train loss: {:.5f}, Val loss: {:.5f}'.format(epoch + 1, loss, val_loss))
        checkpoint(model, val_loss, best, args, epoch)
        best = np.minimum(best, val_loss)


def checkpoint(model, val_loss, best, args, epoch):
    if val_loss < best:
        checkpoint = {'model': model, 'state_dict': model.state_dict()}
        torch.save(checkpoint, './models/' + '__pretrained__' + args.model_id + '.pth')
        print(f'checkpoint saved at epoch {epoch + 1}')