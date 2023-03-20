from typing import Any, List

from autode import Molecule, Atom, Calculation

def orca(
    molecule: Any,
    keywords: List[str],
    conformer_idx: int = 0,
    solvent: str = 'Methanol',
    functional: str = 'B3LYP',
    basis_set: str = '6-31g',
    n_cores: int = 4
):
    orca_method = ORCA()

    keywords = OptKeywords()

    mol = molecule.to_autode_mol(conformer_idx=conformer_idx)
    calc = Calculation(
        name=f'{mol.name}-Calc',
        molecule=mol,
        method=orca_method,
        keywords=keywords,
    )
    calc.run()