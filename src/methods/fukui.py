import os
import subprocess
from typing import Any, List, Optional
import time

from src.utils import read_xyz_file, run_external, run_in_tmp_environment, work_in_tmp_dir

def get_fukui_indices(output, n_atoms):
    lines = output.split('\n')

    start_idx = None
    for idx, line in enumerate(lines):
        if 'f(+)' in line:
            start_idx = idx + 1
            break

    indices = []
    for idx in range(start_idx, start_idx + n_atoms):
        indices.append((
            float(lines[idx].split()[-3]),
            float(lines[idx].split()[-2]),
            float(lines[idx].split()[-1]),
        ))
    return indices
            

def compute_fukui_indices(
    molecule: Any,
    keywords: List[str],
    conformer_idx: int = 0,
    method: str = '2',
    solvent: str = 'Methanol',
    xcontrol_file: Optional[str] = None,
):
    flags = [
        "--chrg",
        str(molecule.charge),
        "--uhf",
        str(molecule.mult),
        "--gfn",
        str(method),
        "--vfukui"
    ]

    if solvent is not None:
        flags += ["--gbsa", solvent]

    if xcontrol_file is not None:
        flags += ["--input", 'mol.input']

    if len(keywords) != 0:
        flags += list(keywords)

    @work_in_tmp_dir(
        filenames_to_copy=[],
        kept_file_exts=(),
    )
    @run_in_tmp_environment(
        OMP_NUM_THREADS=1, 
        MKL_NUM_THREADS=1,
        GFORTRAN_UNBUFFERED_ALL=1
    )
    def execute_xtb():
        if xcontrol_file is not None:
            with open('mol.input', 'w') as f:
                f.write(xcontrol_file)

        file_name = f'mol-{time.time()}.xyz'

        molecule.to_xyz(file_name, conformer_idx)

        cmd = f'{os.environ["XTB_PATH"]} {file_name} {" ".join(flags)}'
        proc = subprocess.Popen(
            cmd.split(), 
            stdout=subprocess.PIPE, 
            stderr=subprocess.DEVNULL, 
            text=True,
        )
        output = proc.communicate()[0]

        indices = get_fukui_indices(output, molecule.n_atoms)
        return indices
    
    indices = execute_xtb()
    return indices


if __name__ == "__main__":
    pass
    # from src.compound import Compound
    # mol = Compound.from_smiles('CC')
    # mol.generate_conformers()

    # indices = compute_fukui_indices(
    #     molecule=mol,
    #     keywords=['--opt'],
    # )
    # print(indices)