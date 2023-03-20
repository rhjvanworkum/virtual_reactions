import pyvista as pv
from typing import List
from ase.data import covalent_radii
from ase.data.colors import cpk_colors
import numpy as np

atom_dict = {
    'H': 1,
    'He': 2,
    'Li': 3,
    'Be': 4,
    'B': 5, 
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9,
    'Ne': 10,
    'Cl': 17,
    'Br': 35,
    'I': 53
}
# cpk_colors = {
#     8: [0.8, 0.0, 0.0],
#     6: [0.5, 0.5 ,0.5],
#     1: [1, 1, 1]
# }

def plot_normal_modes(
    atom_symbols: List[str],
    R: np.array,
    normal_modes: np.array
):
    Z = np.array([atom_dict[atom] for atom in atom_symbols])

    p = pv.Plotter()

    # atoms
    for coords, atom in zip(R, Z):
        p.add_mesh(
            pv.Sphere(
                center=coords,
                radius=covalent_radii[atom],
            ),
            color=cpk_colors[atom],
            show_edges=False,
        )

    # normal mode
    for idx, normal_mode in enumerate(normal_modes):
        p.add_mesh(
            pv.Arrow(
                start=R[idx] + normal_mode * covalent_radii[Z[idx]],
                direction=normal_mode,
                scale=np.linalg.norm(normal_mode) * 5
            ),
            color=[0.5, 0.5, 0.5],
            show_edges=False,
        )

    p.show()