
def method_default_args(dataset):
    defaults = {}

    if dataset == 'DCOPF':
        # dataset related parameters
        defaults['truncate_idx'] = (2,40)
        defaults['primal_const_num'] = 152
        defaults['primal_var_num'] = 161
        defaults['primal_fx_idx'] = (0,48)
        defaults['dual_const_num'] = 49
        defaults['dual_var_num'] = 152
        defaults['dual_fx_idx'] = (0,39)

        # hidden layers related parameters
        defaults['primal_hidden_dim'] = 64
        defaults['primal_hidden_num'] = 1
        defaults['dual_hidden_dim'] = 64
        defaults['dual_hidden_num'] = 1
        defaults['proj_hidden_dim'] = 64
        defaults['proj_hidden_num'] = 2

        # ALM and penalty related parameters
        defaults['penalty_g'] = 50000
        defaults['penalty_h'] = 50000

        # training related parameters
        defaults['loss_type'] = 'obj'
        defaults['optimizer'] = 'Adam'
        defaults['lr'] = 1e-3
        defaults['weight_decay'] = 0.0
        defaults['batch_size'] = 256
        defaults['epochs'] = 20
        defaults['data_generator'] = False
        defaults['self_supervised'] = True

        # evaluation related parameters
        defaults['test_val_train'] = 'val'
        defaults['job'] = 'training'

        # project related parameters
        defaults['max_iter'] = 1000
        defaults['f_tol'] = 1e-9
        defaults['learn2proj'] = False
        defaults['proj_epochs'] = 500

    elif dataset == 'DCOPF_large':
        # dataset related parameters
        defaults['truncate_idx'] = (1,501)
        defaults['primal_const_num'] = 2143
        defaults['primal_var_num'] = 2313
        defaults['primal_fx_idx'] = (0,670)
        defaults['dual_const_num'] = 671
        defaults['dual_var_num'] = 2143
        defaults['dual_fx_idx'] = (0,500)

        # hidden layers related parameters
        defaults['primal_hidden_dim'] = 1024
        defaults['primal_hidden_num'] = 1
        defaults['dual_hidden_dim'] = 1024
        defaults['dual_hidden_num'] = 1
        defaults['proj_hidden_dim'] = 2048
        defaults['proj_hidden_num'] = 2

        # ALM and penalty related parameters
        defaults['penalty_g'] = 50
        defaults['penalty_h'] = 50

        # training related parameters
        defaults['loss_type'] = 'obj'
        defaults['optimizer'] = 'Adam'
        defaults['lr'] = 1e-3
        defaults['weight_decay'] = 0.0
        defaults['batch_size'] = 256
        defaults['epochs'] = 20
        defaults['data_generator'] = False
        defaults['self_supervised'] = True

        # evaluation related parameters
        defaults['test_val_train'] = 'val'
        defaults['job'] = 'training'

        # project related parameters
        defaults['max_iter'] = 1000
        defaults['f_tol'] = 1e-12
        defaults['learn2proj'] = False
        defaults['proj_epochs'] = 500

    else:
        raise NotImplementedError

    return defaults