from utils import *
import torch
import csv
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import random
import time
import os
import pickle


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)


def self_supervised_generator(args, data, num_points, random_range):
    b_ref = data['b_ref']
    data_points = []
    for _ in range(num_points):
        # random_number = random_range[0] + torch.rand(b_ref.size()) * (random_range[1] - random_range[0])
        random_number = random.uniform(random_range[0], random_range[1])
        x = b_ref * random_number
        data_points.append(x)
    input_train = torch.stack(data_points)
    train_shape_in = input_train.shape
    train_shape_out = num_points
    target_train = torch.zeros(train_shape_out)
    mean = input_train.mean()
    std = input_train.std()

    train_data = TensorDataset(input_train, target_train)
    train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    return train, mean, std, train_shape_in, train_shape_out


def run_training(args, data, problem):
    model = load_model(args, problem)
    print(f'----- {args.model_id} in {args.dataset} dataset -----')
    print('#params:', sum(p.numel() for p in model.parameters()))
    optimizer = get_optimizer(args, model)

    print('loss_type: ', args.loss_type)

    if args.data_generator:
        print('self-generated data training code has not been implemented yet')
    else:
        print('training dataset is given already')

    start_time = time.time()
    ##############################################################################################################
    Learning(args, data, problem, model, optimizer)
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


def Learning(args, data, problem, model, optimizer):
    best = np.inf
    stats = {}
    for epoch in range(args.epochs):
        epoch_stats = {}
        # train
        model.train()
        for (inputs, targets) in data['train']:
            inputs, targets = process_for_training(inputs, targets, args)
            optimizer_step(model, optimizer, inputs, targets, args, data, problem, epoch_stats)
        # validate
        model.eval()
        validate_model(model, args, data, problem, epoch_stats)
        # model checkpoint
        checkpoint(model, best, args, epoch_stats, epoch)
        best = np.minimum(best, np.mean(epoch_stats['val_gap_mean']))
        # print epoch_stats
        print('----- Epoch {} -----'.format(epoch))
        print('Train loss: {:.5f}, Train time: {: .5f}'.format(np.mean(epoch_stats['train_loss']),
                                                               epoch_stats['train_time']))
        print('eq mean: {: .5f}, eq max: {: .5f}, eq worst: {: .5f}'.format(np.mean(epoch_stats['train_eq_mean']),
                                                                             np.mean(epoch_stats['train_eq_max']),
                                                                             np.max(epoch_stats['train_eq_worst'])))
        print('ineq mean: {: .5f}, ineq max: {: .5f}, ineq worst: {: .5f}'.format(np.mean(epoch_stats['train_ineq_mean']),
                                                                                  np.mean(epoch_stats['train_ineq_max']),
                                                                                  np.max(epoch_stats['train_ineq_worst'])))
        print('Val gap mean: {:.5f}, val gap worst: {: .5f}'.format(np.mean(epoch_stats['val_gap_mean']),
                                                                    np.max(epoch_stats['val_gap_worst'])))
        print('eq mean: {: .5f}, eq max: {: .5f}, eq worst: {: .5f}'.format(np.mean(epoch_stats['val_eq_mean']),
                                                                            np.mean(epoch_stats['val_eq_max']),
                                                                            np.max(epoch_stats['val_eq_worst'])))
        print('ineq mean: {: .5f}, ineq max: {: .5f}, ineq worst: {: .5f}'.format(np.mean(epoch_stats['val_ineq_mean']),
                                                                                  np.mean(epoch_stats['val_ineq_max']),
                                                                                  np.max(epoch_stats['val_ineq_worst'])))
        print('Infer time (batched inference mean): {: .5f}'.format(np.mean(epoch_stats['val_time'])))
        print('Val projections: {: 0f}'.format(np.mean(epoch_stats['val_proj'])))
        # save stats
        if args.saveAllStats:
            if epoch == 0:
                for key in epoch_stats.keys():
                    stats[key] = np.expand_dims(np.array(epoch_stats[key]), axis=0)
            else:
                for key in epoch_stats.keys():
                    stats[key] = np.concatenate((stats[key], np.expand_dims(np.array(epoch_stats[key]), axis=0)))
        else:
            stats = epoch_stats

        if epoch % args.resultSaveFreq == 0:
            with open(f'./logs/{args.model_id}_TrainingStats.dict', 'wb') as f:
                pickle.dump(stats, f)


def optimizer_step(model, optimizer, inputs, targets, args, data, problem, epoch_stats):
    start_time = time.time()
    optimizer.zero_grad()
    z_star, z1 = model(inputs)
    train_loss = get_loss(z_star, z1, targets, inputs, problem, args, args.loss_type)  #todo, item()?
    train_loss.backward()
    optimizer.step()
    train_time = time.time() - start_time

    dict_agg(epoch_stats, 'train_time', train_time, op='sum')
    dict_agg(epoch_stats, 'train_loss', train_loss.detach().cpu().numpy())

    violation_agg(z_star, inputs, problem, epoch_stats, 'train')

    # print(epoch_stats)


def validate_model(model, args, data, problem, epoch_stats):
    for i, (inputs, targets) in enumerate(data['val']):
        inputs, targets = process_for_training(inputs, targets, args)
        start_time = time.time()
        z_star, z1, proj_num = model(inputs)
        val_time = time.time() - start_time
        predicted_obj = problem.obj_fn(z_star, inputs)
        optimality_gap = (predicted_obj - targets) / targets
        gap_mean, gap_worst = get_gap_mean_worst(optimality_gap)

        # print(f'shape of gap_mean: {gap_mean.shape}, shape of gap_worst: {gap_worst.shape}')
        # print(f'gap_mean: {type(gap_mean)}, gap_worst: {gap_worst}')
        # print(f'gap_mean.detach().cpu().numpy() {type(gap_mean.detach().cpu().numpy())}')
        # print(f'gap_worst.detach().cpu().numpy() {gap_worst.detach().cpu().numpy()}')

        dict_agg(epoch_stats, 'val_time', np.array([val_time]))
        dict_agg(epoch_stats, 'val_proj', np.array([proj_num]))
        dict_agg(epoch_stats, 'val_gap_mean', gap_mean.detach().cpu().numpy())
        dict_agg(epoch_stats, 'val_gap_worst', gap_worst.detach().cpu().numpy())

        violation_agg(z_star, inputs, problem, epoch_stats, 'val')


def dict_agg(stats, key, value, op='concat'):
    if key in stats.keys():
        if op == 'sum':
            stats[key] += value
        elif op == 'concat':
            if value.ndim == 0:
                value = value.reshape(1)
            stats[key] = np.concatenate((stats[key], value), axis=0)
        else:
            raise NotImplementedError
    else:
        if op == 'concat':
            if value.ndim == 0:
                value = value.reshape(1)
        stats[key] = value


def violation_agg(z_star, inputs, problem, epoch_stats, prefix):
    with torch.no_grad():
        eq_residual = problem.eq_residual(z_star, inputs)
        ineq_residual = problem.ineq_residual(z_star)
        eq_mean, eq_max, eq_worst = get_violation_mean_max_worst(eq_residual)
        ineq_mean, ineq_max, ineq_worst = get_violation_mean_max_worst(ineq_residual)

        dict_agg(epoch_stats, f'{prefix}_eq_mean', eq_mean.detach().cpu().numpy())
        dict_agg(epoch_stats, f'{prefix}_eq_max', eq_max.detach().cpu().numpy())
        dict_agg(epoch_stats, f'{prefix}_eq_worst', eq_worst.detach().cpu().numpy())
        dict_agg(epoch_stats, f'{prefix}_ineq_mean', ineq_mean.detach().cpu().numpy())
        dict_agg(epoch_stats, f'{prefix}_ineq_max', ineq_max.detach().cpu().numpy())
        dict_agg(epoch_stats, f'{prefix}_ineq_worst', ineq_worst.detach().cpu().numpy())


def checkpoint(model, best, args, epoch_stats, epoch):
    gap_mean = np.mean(epoch_stats['val_gap_mean'])
    if gap_mean < best:
        checkpoint = {'model': model, 'state_dict': model.state_dict()}
        torch.save(checkpoint, './models/' + args.model_id + '.pth')
        print(f'checkpoint saved at epoch {epoch}')


def evaluate_model(args, data, problem):
    test_stats = {}
    model = load_model(args, problem)
    load_weights(model, args.model_id)
    model.eval()
    model.report_projection = True
    for i, (inputs, targets) in enumerate(data['val']):
        inputs, targets = process_for_training(inputs, targets, args)
        start_time = time.time()
        z_star, z1, proj_num = model(inputs)
        test_time = time.time() - start_time
        predicted_obj = problem.obj_fn(z_star, inputs)
        optimality_gap = (predicted_obj - targets) / targets
        gap_mean, gap_worst = get_gap_mean_worst(optimality_gap)

        dict_agg(test_stats, 'test_time', np.array([test_time]))
        dict_agg(test_stats, 'test_proj', np.array([proj_num]))
        dict_agg(test_stats, 'test_gap_mean', gap_mean.detach().cpu().numpy())
        dict_agg(test_stats, 'test_gap_worst', gap_worst.detach().cpu().numpy())

        violation_agg(z_star, inputs, problem, test_stats, 'test')

    with open(f'./logs/{args.model_id}_TestStats.dict', 'wb') as f:
        pickle.dump(test_stats, f)

    calculate_scores(args, data)


def calculate_scores(args, data):
    if os.path.exists(f'./logs/{args.model_id}_TrainingStats.dict'):
        try:
            with open(f'./logs/{args.model_id}_TrainingStats.dict', 'rb') as f:
                training_stats = pickle.load(f)
        except:
            print(f'{args.model_id}_TrainingStats.dict is missing. Load test stats only.')

    with open(f'./logs/{args.model_id}_TestStats.dict', 'rb') as f:
        test_stats = pickle.load(f)

    if args.self_supervised:
        extension = 'self_'
    else:
        extension = ''
    target_val = torch.load(
        './data/' + args.dataset + '/' + args.test_val_train + '/' + extension + 'target_' + args.test_val_train + '.pt')

    scores = {'max_obj_true': target_val.max(),
              'min_obj_true': target_val.min(),
              'test_optimality_gap_mean': np.mean(test_stats['test_gap_mean']),
              'test_optimality_gap_worst': np.max(test_stats['test_gap_worst']),
              'test_eq_mean': np.mean(test_stats['test_eq_mean']),
              'test_eq_max': np.mean(test_stats['test_eq_max']),
              'test_eq_worst': np.max(test_stats['test_eq_worst']),
              'test_ineq_mean': np.mean(test_stats['test_ineq_mean']),
              'test_ineq_max': np.mean(test_stats['test_ineq_max']),
              'test_ineq_worst': np.max(test_stats['test_ineq_worst']),
              'train_time': np.sum(training_stats['train_time']),
              'val_time': np.mean(training_stats['val_time']),
              'test_time': np.mean(test_stats['test_time']),
              'val_proj': np.mean(training_stats['val_proj']),
              'test_proj': np.mean(test_stats['test_proj'])}
    print(scores)
    create_report(scores, args)


def create_report(scores, args):
    args_dict = args_to_dict(args)
    # combine scores and args dict
    args_scores_dict = args_dict | scores
    # save dict
    save_dict(args_scores_dict, args)


def args_to_dict(args):
    return vars(args)


def save_dict(dictionary, args):
    w = csv.writer(open('./data/results_summary/' + args.model_id + '.csv', 'w'))
    # loop over dictionary keys and values
    for key, val in dictionary.items():
        # write every key and value to file
        w.writerow([key, val])


def load_weights(model, model_id):
    PATH = './models/' + model_id + '.pth'
    checkpoint = torch.load(PATH, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    return model


