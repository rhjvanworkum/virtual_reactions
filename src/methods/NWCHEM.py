import os
import shutil
import subprocess
from typing import Any, List, Optional
import time

from autode.utils import run_external_monitored
from src.utils import Atom, work_in_tmp_dir, write_xyz_file

def energy_from(lines):
    wf_strings = [
        "Total CCSD energy",
        "Total CCSD(T) energy",
        "Total SCS-MP2 energy",
        "Total MP2 energy",
        "Total RI-MP2 energy",
    ]

    for line in reversed(lines):
        if any(
            string in line
            for string in ["Total DFT energy", "Total SCF energy"]
        ):
            return line.split()[4]
        if any(string in line for string in wf_strings):
            return line.split()[3]

def coordinates_from(lines):

    xyzs_section = False
    geometry = []

    for line in lines:
        if "Output coordinates in angstroms" in line:
            xyzs_section = True
            geometry.clear()

        if "Atomic Mass" in line:
            xyzs_section = False

        if xyzs_section and len(line.split()) == 6:
            if line.split()[0].isdigit():
                _, symbol, _, x, y, z = line.split()
                geometry.append(Atom(symbol, float(x), float(y), float(z)))

    return geometry

def nwchem(
    molecule: Any,
    keywords: List[str],
    conformer_idx: int = 0,
    solvent: str = 'Methanol',
    functional: str = 'B3LYP',
    basis_set: str = '6-31g',
    n_cores: int = 4
):
    @work_in_tmp_dir(
        filenames_to_copy=[],
        kept_file_exts=(),
    )
    def execute_nwchem():
        with open('test.in', "w") as inp_file:
            print(f"start Calculation\necho", file=inp_file)
            # solvent
            if solvent is not None:
                print(
                    f"cosmo\n "
                    f"do_cosmo_smd true\n "
                    f"solvent {solvent}\n"
                    f"end",
                    file=inp_file,
                )
            # geometry
            print("geometry noautosym", end=" ", file=inp_file)
            print("", file=inp_file)
            for atom in molecule.conformers[conformer_idx]:
                print(
                    f"{atom.atomic_symbol:<3} {atom.x:^12.8f} {atom.y:^12.8f} {atom.z:^12.8f}",
                    file=inp_file,
                )
            print("end", file=inp_file)

            print(f"charge {molecule.charge}", file=inp_file)
            print(f'memory {int(4096)} mb', file=inp_file)
            print(f"dft\n  maxiter 100\n  xc {functional}\nend", file=inp_file)
            print(f"basis\n  *   library {basis_set}\nend", file=inp_file)
            if "--opt" in keywords:
                print(f"task dft optimize", file=inp_file)
            else:
                print(f"task dft", file=inp_file)


        params = [
            "mpirun",
            "-np",
            str(n_cores),
            "nwchem",
            'test.in',
        ]

        run_external_monitored(
            params,
            'test.out',
            break_words=["Received an Error", "MPI_ABORT"],
        )

        with open('test.out', 'r') as f:
            lines = f.readlines()

        energy = energy_from(lines)
        if '--opt' in keywords:
            geom = coordinates_from(lines)
        else:
            geom = None

        return energy, geom

    energy, final_geometry = execute_nwchem()
    return energy, final_geometry 

if __name__ == "__main__":
    from src.compound import Compound
    mol = Compound.from_smiles('CC')
    mol.generate_conformers()

    e, geom = nwchem(
        molecule=mol,
        keywords=['--opt'],
    )
    print(e, geom)