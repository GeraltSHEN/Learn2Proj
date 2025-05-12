import yaml


def get_default_args(dataset):
    defaults = {}

    defaults["model"] = "mlp"
    defaults["algo"] = "LDRPM"  # LDRPM, OPTNET, POCS
    defaults["dc3_lr"] = 1e-7
    defaults["dc3_momentum"] = 0.5
    defaults["dc3_softweighteqfrac"] = 0.5
    defaults["dc3_softweight"] = 10
    defaults['ldr_temp'] = 10
    # dataset related parameters
    defaults["data_generator"] = True
    defaults["renew_freq"] = 20
    # layers related parameters
    defaults["hidden_dims"] = [256, 256]
    defaults["act_cls"] = "relu"
    defaults["batch_norm"] = True
    # training related parameters
    defaults["loss_type"] = "obj"
    defaults["optimizer"] = "adam"
    defaults["lr"] = 0.0001
    defaults["weight_decay"] = 1e-8
    defaults["batch_size"] = 64
    defaults["bsz_factor"] = 20
    defaults["epochs"] = 300
    defaults["pretrain_epochs"] = 50
    defaults["alpha_penalty"] = 0

    # if dataset == 'SSLdebug':
    #     defaults["batch_size"] = 4
    #     defaults["bsz_factor"] = 5
    #     defaults["epochs"] = 4

    # elif dataset == 'case14_ieee':
    #     defaults["eq_tol"] = 1e-4
    #     defaults["ineq_tol"] = 1e-4
    #     defaults["max_iters"] = 300
    #     # layers related parameters
    #     defaults["hidden_dims"] = [256, 256]
    #     # training related parameters
    #     defaults["batch_size"] = 64
    #     defaults["bsz_factor"] = 100
    #     defaults["epochs"] = 1000
    #
    # elif dataset == 'case30_ieee':
    #     defaults["eq_tol"] = 1e-4
    #     defaults["ineq_tol"] = 1e-4
    #     defaults["max_iters"] = 300
    #     # layers related parameters
    #     defaults["hidden_dims"] = [256, 256]
    #     # training related parameters
    #     defaults["batch_size"] = 64
    #     defaults["bsz_factor"] = 100
    #     defaults["epochs"] = 1000
    #
    # elif dataset == 'case57_ieee':
    #     defaults["eq_tol"] = 1e-4
    #     defaults["ineq_tol"] = 1e-4
    #     defaults["max_iters"] = 300
    #     # layers related parameters
    #     defaults["hidden_dims"] = [256, 256]
    #     # training related parameters
    #     defaults["batch_size"] = 64
    #     defaults["bsz_factor"] = 100
    #     defaults["epochs"] = 1000
    #
    # elif dataset == 'case118_ieee':
    #     defaults["eq_tol"] = 1e-4
    #     defaults["ineq_tol"] = 1e-4
    #     defaults["max_iters"] = 300
    #     # layers related parameters
    #     defaults["hidden_dims"] = [256, 256]
    #     # training related parameters
    #     defaults["batch_size"] = 64
    #     defaults["bsz_factor"] = 100
    #     defaults["epochs"] = 1000
    #
    # elif dataset == 'case200_activ':
    #     defaults["eq_tol"] = 1e-4
    #     defaults["ineq_tol"] = 1e-4
    #     defaults["max_iters"] = 300
    #     # layers related parameters
    #     defaults["hidden_dims"] = [256, 256]
    #     # training related parameters
    #     defaults["batch_size"] = 64
    #     defaults["bsz_factor"] = 100
    #     defaults["epochs"] = 1000

    # else:
    #     raise NotImplementedError

    with open(f"./cfg/{dataset}_0", "w") as yaml_file:
        yaml.dump(defaults, yaml_file, default_flow_style=False)

    print(f"Default Configuration file saved to ./cfg/{dataset}_0")

# SSLdebug, case14_ieee, case30_ieee, case57_ieee, case118_ieee, case200_activ
get_default_args("SSLdebug")

for dataset in ['case14_ieee', 'case30_ieee', 'case57_ieee', 'case118_ieee', 'case200_activ']:
    get_default_args(dataset)

