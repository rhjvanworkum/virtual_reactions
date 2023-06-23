import yaml

with open(f'./data/ani-cxx/experiment_2/splits/split.yaml', "r") as f:
    split = yaml.load(f, Loader=yaml.Loader)

total_count = 0
for mol_idx in split['train'].keys():
    print(len(split['train'][mol_idx]['cc']))
    total_count += len(split['train'][mol_idx]['cc'])

print(total_count)