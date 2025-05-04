import yaml


def get_default_args(dataset):
    defaults = {}

    if dataset == 'SSLdebug':
        defaults["model"] = "mlp"
        # dataset related parameters
        defaults["data_generator"] = True
        defaults["renew_freq"] = 88
        # layers related parameters
        defaults["hidden_dims"] = [256, 256]
        defaults["act_cls"] = "relu"
        defaults["batch_norm"] = True
        # training related parameters
        defaults["loss_type"] = "obj"
        defaults["optimizer"] = "adam"
        defaults["lr"] = 0.0001
        defaults["weight_decay"] = 1e-8
        defaults["batch_size"] = 4
        defaults["bsz_factor"] = 5
        defaults["epochs"] = 4
    else:
        raise NotImplementedError

    with open(f"./cfg/{dataset}_0", "w") as yaml_file:
        yaml.dump(defaults, yaml_file, default_flow_style=False)

    print(f"Default Configuration file saved to ./cfg/{dataset}_0")


get_default_args("SSLdebug")

