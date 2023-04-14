from concurrent.futures import ProcessPoolExecutor
import autode as ade
from autode.wrappers.XTB import XTB
from autode.species import Complex
from autode.species.molecule import Molecule

from typing import Any
import numpy as np

def write_xyz_file(atoms: Any, filename: str):
  with open(filename, 'w') as f:
    f.write(str(len(atoms)) + ' \n')
    f.write('\n')

    for atom in atoms:
        f.write(atom.atomic_symbol)
        for cartesian in ['x', 'y', 'z']:
            if getattr(atom.coord, cartesian) < 0:
                f.write('         ')
            else:
                f.write('          ')
            f.write("%.5f" % getattr(atom.coord, cartesian))
        f.write('\n')
    
    f.write('\n')


if __name__ == "__main__":
    ade.Config.n_cores = 4
    ade.Config.XTB.path = '/home/rhjvanworkum/xtb-6.5.1/bin/xtb'
    np.random.seed(42)
    n_conformers = 10
    method = XTB()

    n_processes = 10

    catalyst = Molecule(smiles="[Pd]([P](c1ccccc1)(c1ccccc1)(c1ccccc1))[P](c1ccccc1)(c1ccccc1)(c1ccccc1)")
    catalyst._generate_conformers(n_conformers)

    mol = Molecule(smiles="c1ccccc1Br")
    mol._generate_conformers(n_conformers)

    complex = Complex(mol, catalyst)
    complex._generate_conformers()

    def optimise_conformer(conformer):
        conformer.optimise(method=method)

    args = [conformer for conformer in complex.conformers]
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        results = executor.map(optimise_conformer, args)

    distances = []
    for conformer in complex.conformers:
        for atom in conformer.atoms:
            if atom.atomic_symbol == 'Pd':
                pd_pos = atom.coord
                break
          
        for atom in conformer.atoms:
            if atom.atomic_symbol == 'Br':
                br_pos = atom.coord
                break

        distances.append(
            np.linalg.norm(br_pos - pd_pos)
        )
    
    write_xyz_file(complex.conformers[np.argmin(distances)].atoms, 'test.xyz')