from typing import Any, List

from pyscf import gto
from pyscf.geomopt.geometric_solver import optimize

from src.utils import Atom

SOLVENT_CONSTANT_DICT = {
    'Methanol': 32.613
}

def mol_to_pyscf_mol(
    atoms: List[Atom],
    charge: int,
    mult: int,
    basis_set: str = 'sto_6g'
):
    atoms_string = " \n".join([
        f"{atom.atomic_symbol} {atom.x} {atom.y} {atom.z}" for atom in atoms
    ])
    
    mol = gto.M(
        atom=atoms_string,
        basis=basis_set,
        verbose=0,
        charge=charge,
        spin=mult
    )

    return mol

def pyscf(
    molecule: Any,
    keywords: List[str],
    conformer_idx: int = 0,
    solvent: str = 'Methanol',
    functional: str = 'B3LYP',
    basis_set: str = '6-31g',
    n_cores: int = 4
):
    mol = mol_to_pyscf_mol(
        atoms=molecule.conformers[conformer_idx],
        basis_set=basis_set,
        charge=molecule.charge,
        mult=molecule.mult
    )

    mf = mol.RKS(xc=functional)
    if solvent is not None:
        mf = mf.DDCOSMO()
        mf.with_solvent.eps = SOLVENT_CONSTANT_DICT[solvent]
    
    if '--opt' in keywords:
        mol_eq = optimize(mf, maxsteps=250, verbose=0)
        energy = mf.e_tot
        geometry = [
            Atom(symbol, float(coord[0]), float(coord[1]), float(coord[2])) for symbol, coord in zip(mol_eq.elements, mol_eq.atom_coords())
        ]
    else:
        mf.run()
        energy = mf.e_tot
        geometry = None

    return energy, geometry



