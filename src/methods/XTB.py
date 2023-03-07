import os
import shutil
import subprocess
from typing import Any, List, Optional
import time


from src.utils import read_xyz_file, run_external, run_in_tmp_environment, work_in_tmp_dir

def run_xtb(args):
    (molecule, keywords, conformer_idx, method, solvent, xcontrol_file) = args
    return xtb(molecule, keywords, conformer_idx, method, solvent, xcontrol_file)

def xtb(
    molecule: Any,
    keywords: List[str],
    conformer_idx: int = 0,
    method: str = '1',
    solvent: str = 'Methanol',
    xcontrol_file: Optional[str] = None,
    comment: Optional[str] = None
):
    flags = [
        "--chrg",
        str(molecule.charge),
        "--uhf",
        str(molecule.mult),
        "--gfn",
        str(method)
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

        energy = None
        for line in reversed(output.split('\n')):
            if "total E" in line:
                energy = float(line.split()[-1])
            if "TOTAL ENERGY" in line:
                energy = float(line.split()[-3])

        if '--opt' in keywords and os.path.exists('xtbopt.xyz'):
            final_geometry = read_xyz_file('xtbopt.xyz')
        else:
            final_geometry = None

        return energy, final_geometry
    
    energy, final_geometry = execute_xtb()
    return energy, final_geometry 

if __name__ == "__main__":
    from src.compound import Compound
    mol = Compound('CC')
    mol.generate_conformers()

    e, geom = xtb(
        molecule=mol,
        keywords=['--opt'],
    )
    print(e, geom)