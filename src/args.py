import yaml
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str)
    parser.add_argument("--project", type=str)
    parser.add_argument("--use_wandb", type=bool)

    parser.add_argument("--n_replications", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--use_features", type=bool)

    parser.add_argument("--dataset_type", type=str)
    parser.add_argument("--folder_path", type=str)
    parser.add_argument("--simulation_type", type=str)

    parser.add_argument("--split_type", type=str)
    parser.add_argument("--transductive", type=bool)
    parser.add_argument("--clustering_method", type=str)
    parser.add_argument("--min_cluster_size", type=int)
    parser.add_argument("--n_ood_test_compounds", type=int)

    return parser.parse_args()

def parse_config_from_command_line():
    with open('config.yaml', "r") as f:
        config = yaml.safe_load(f)
    
    args = parse_arguments()

    for arg in config.keys():
        if getattr(args, arg):
            config[arg] = getattr(args, arg)

    return config