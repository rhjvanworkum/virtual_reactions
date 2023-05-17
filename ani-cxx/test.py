import yaml
from ase.db import connect

db_path = './data/experiment_2/dataset.db'

with open(f'data/experiment_2/splits/split.yaml', "r") as f:
    split = yaml.load(f, Loader=yaml.Loader)


for i in split['train'].keys():
    idx = split['train'][i]['dft'][0]

    with connect(db_path) as conn:
        print(conn.get(idx + 1).numbers)
