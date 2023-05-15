from typing import Any, List, Literal

import autode as ade
from autode.utils import work_in_tmp_dir
from autode.wrappers.ORCA import ORCA

def orca(
    molecule: Any,
    job: Literal["sp", "opt"] = "sp",
    conformer_idx: int = 0,
    functional: str = 'B3LYP',
    basis_set: str = 'def2-SVP',
    n_cores: int = 1
):
    ade.Config.n_cores = n_cores
    ade.Config.ORCA.path = "/home/rhjvanworkum/orca/orca"
    ade.Config.ORCA.keywords.sp = [
        functional, basis_set
    ]   
    ade.Config.ORCA.keywords.opt = [
        functional, basis_set
    ]

    @work_in_tmp_dir(
        filenames_to_copy=[],
        kept_file_exts=(),
    )
    def compute():
        ade_species = molecule.to_autode_mol(conformer_idx=conformer_idx)
        
        if job == "sp":
            ade_species.single_point(method=ORCA())
            return ade_species.energy, ade_species.partial_charges
        elif job == "opt":
            ade_species.optimise(method=ORCA())
            return ade_species.coordinates

    
    return compute()