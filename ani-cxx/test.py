from ase.db import connect


with connect('10k_dataset.db') as conn:
    print(conn)