import yaml
import argparse

def parse_chemprop_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str)
    parser.add_argument("--project", type=str)
    parser.add_argument("--use_wandb", type=bool)

    parser.add_argument("--n_replications", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--use_features", type=bool)
    parser.add_argument("--ensemble_size", type=int)

    parser.add_argument("--dataset_type", type=str)
    parser.add_argument("--folder_path", type=str)
    parser.add_argument("--simulation_type", type=str)

    parser.add_argument("--train_subset", type=float)
    parser.add_argument("--ood_subset", type=float)
    parser.add_argument("--split_type", type=str)
    parser.add_argument("--transductive", type=bool)
    parser.add_argument("--clustering_method", type=str)
    parser.add_argument("--min_cluster_size", type=int)
    parser.add_argument("--n_ood_test_compounds", type=int)

    return parser.parse_args()

def parse_chemprop_config_from_command_line():
    with open('chemprop_config.yaml', "r") as f:
        config = yaml.safe_load(f)
    
    args = parse_chemprop_arguments()

    for arg in config.keys():
        if getattr(args, arg):
            config[arg] = getattr(args, arg)

    return config



def parse_schnet_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str)
    parser.add_argument("--project", type=str)
    parser.add_argument("--use_wandb", type=bool)
    parser.add_argument("--n_devices", type=int)

    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--models_path", type=str)
    parser.add_argument("--split_file_path", type=str)
    parser.add_argument("--has_virtual_reactions", type=bool)
    parser.add_argument("--n_simulations", type=int)

    parser.add_argument("--epochs", type=int)
    parser.add_argument("--n_replications", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--cutoff", type=float)
    parser.add_argument("--n_radial", type=int)
    parser.add_argument("--n_atom_basis", type=int)
    parser.add_argument("--n_interactions", type=int)
    parser.add_argument("--sim_embedding_dim", type=int)

    return parser.parse_args()

def parse_schnet_config_from_command_line():
    with open('schnet_config.yaml', "r") as f:
        config = yaml.safe_load(f)
    
    args = parse_schnet_arguments()

    for arg in config.keys():
        if getattr(args, arg):
            config[arg] = getattr(args, arg)

    return config