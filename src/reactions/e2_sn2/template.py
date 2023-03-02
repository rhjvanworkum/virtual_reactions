from typing import Callable, List
import autode as ade
from autode.values import Coordinates
from autode.bond_rearrangement import BondRearrangement
from autode.species.complex import Complex

from scipy import spatial
import numpy as np
from copy import copy

from src.utils import translate_rotate_reactant, write_xyz_file


class E2Sn2ReactionIndices:
    def __init__(
        self,
        leaving_group_idx: int,
        central_atom_idx: int,
        attacked_atom_idx: int,
        nucleophile_idx: int
    ) -> None:
        self.leaving_group_idx = leaving_group_idx
        self.central_atom_idx = central_atom_idx
        self.attacked_atom_idx = attacked_atom_idx
        self.nucleophile_idx = nucleophile_idx


class Sn2ReactionComplexTemplate:
    dist_dict = {
        'H': 1.14,
        'F': 1.41,
        'Cl': 1.86,
        'Br': 2.04
    }
    hydrogen_dist_dict = {
        'H': 0.78,
        'F': 0.96,
        'Cl': 1.33,
        'Br': 2.48
    }

    def __init__(
        self,
        checks: List[Callable] = []
    ) -> None:
        super(Sn2ReactionComplexTemplate, self).__init__()
        self.checks = checks

    def _construct_bond_rearrangement(
        self,
        indices: E2Sn2ReactionIndices
    ):
        bond_rearr = BondRearrangement(
            forming_bonds=[(indices.central_atom_idx, indices.nucleophile_idx)],
            breaking_bonds=[(indices.central_atom_idx, indices.leaving_group_idx)]
        )
        return bond_rearr


    def generate_reaction_complex(
        self,
        substrate: ade.Species,
        nucleophile: ade.Species,
        indices: E2Sn2ReactionIndices,
        method: ade.methods
    ) -> ade.Species:   
        """
        Tries to generate a reaction complex
        """
        # generate coordinates along substitution center
        diff_vec = substrate.atoms[indices.leaving_group_idx].coord - substrate.atoms[indices.central_atom_idx].coord
        diff_vec /= np.linalg.norm(diff_vec)
        rot_vec = substrate.atoms[indices.central_atom_idx].coord - substrate.atoms[indices.attacked_atom_idx].coord
        rot_vec = rot_vec / np.linalg.norm(rot_vec) * np.radians(180)
        r = spatial.transform.Rotation.from_rotvec(rot_vec)
        distance = self.dist_dict[nucleophile.atoms[0].atomic_symbol] + 1.0
        nucleophile_coords = substrate.atoms[indices.central_atom_idx].coord + r.apply(distance * diff_vec)
        
        # generate constraints + bond_rearrangements
        constraints = {
            (indices.central_atom_idx, len(substrate.atoms)): distance
        }
        tmp_indices = copy(indices)
        tmp_indices.nucleophile_idx = len(substrate.atoms)
        bond_rearrangement = self._construct_bond_rearrangement(tmp_indices)

        # generate reaction complex
        reaction_complex = substrate.copy()
        new_atoms = substrate.atoms + [ade.Atom(
                atomic_symbol=nucleophile.atoms[0].atomic_symbol,
                x=nucleophile_coords[0],
                y=nucleophile_coords[1],
                z=nucleophile_coords[2]
            )]
        reaction_complex._parent_atoms = [atom for atom in new_atoms]
        reaction_complex._coordinates = Coordinates(np.array([atom.coord for atom in new_atoms]))
        reaction_complex.charge = -1

        # constrained optimization of reaction complex
        reaction_complex.constraints.distance = constraints
        reaction_complex.optimise(method=method)

        # relaxed optimization of reaction complex
        reaction_complex.constraints.distance = {}
        reaction_complex.optimise(method=method)

        # check conditions
        conditions_passed = []
        for check in self.checks:
            conditions_passed.append(
                check(
                    reaction_complex,
                    indices,
                    self.dist_dict,
                    self.hydrogen_dist_dict,
                    nucleophile.atoms[0].atomic_symbol
                )
            )
        
        return (False not in conditions_passed), reaction_complex, bond_rearrangement

def get_hs(
    reaction_complex: ade.Species, 
    indices: E2Sn2ReactionIndices
) -> List[int]:
    total_hs = []

    for idx, atom in enumerate(reaction_complex.atoms):
        if atom.atomic_symbol == 'H':
            total_hs.append((idx, atom))

    hs = []
    for (idx, atom) in total_hs:
        if np.linalg.norm(reaction_complex.atoms[indices.attacked_atom_idx].coord - atom.coord) < 1.2:
            hs.append(idx)

    if len(hs) > 3:
        print('C has more than 3 hydrogen bonded to them!')
        hs = []
        for (idx, atom) in total_hs:
            if np.linalg.norm(reaction_complex.atoms[indices.attacked_atom_idx].coord - atom.coord) < 1.1:
                hs.append(idx)

        if len(hs) > 3:
            hs = []
            for (idx, atom) in total_hs:
                if np.linalg.norm(reaction_complex.atoms[indices.attacked_atom_idx].coord - atom.coord) < 1.0:
                    hs.append(idx)

    return hs

def get_opposite_hydrogen(
    reaction_complex: ade.Species,
    indices: E2Sn2ReactionIndices,
    hs_indices: List[int]
) -> int:
    leaving_group_vec = reaction_complex.atoms[indices.leaving_group_idx].coord - reaction_complex.atoms[indices.central_atom_idx].coord
    leaving_group_vec /= np.linalg.norm(leaving_group_vec)

    angles = []
    for h_idx in hs_indices:
        vec = reaction_complex.atoms[h_idx].coord - reaction_complex.atoms[indices.attacked_atom_idx].coord
        vec /= np.linalg.norm(vec)
        angle = np.arccos(np.dot(leaving_group_vec, vec))
        angles.append(angle)
    
    return hs_indices[np.argmin(np.array(angles))]


class E2ReactionComplexTemplate:

    dist_dict = {
        'H': 1.14,
        'F': 1.41,
        'Cl': 1.86,
        'Br': 2.04
    }
    hydrogen_dist_dict = {
        'H': 0.78,
        'F': 0.96,
        'Cl': 1.33,
        'Br': 2.48
    }

    def __init__(
        self,
        checks: List[Callable] = []
    ) -> None:
        super(E2ReactionComplexTemplate, self).__init__()
        self.checks = checks

        self.h_idx = None
        self.indices = None

    def _construct_bond_rearrangement(
        self,
        h_idx: int,
        indices: E2Sn2ReactionIndices
    ):
        bond_rearr = BondRearrangement(
            forming_bonds=[(h_idx, indices.nucleophile_idx)],
            breaking_bonds=[(h_idx, indices.attacked_atom_idx), (indices.central_atom_idx, indices.leaving_group_idx)]
        )
        return bond_rearr

    def generate_reaction_complex(
        self,
        substrate: ade.Species,
        nucleophile: ade.Species,
        indices: E2Sn2ReactionIndices,
        method: ade.methods
    ) -> ade.Species:   
        """
        Tries to generate a reaction complex
        """
        # get attacking H index
        hs_indices = get_hs(substrate, indices)
        h_idx = get_opposite_hydrogen(substrate, indices, hs_indices)

        # set nucleophile coords
        distance = self.dist_dict[nucleophile.atoms[0].atomic_symbol]
        diff_vec = substrate.atoms[h_idx].coord - substrate.atoms[indices.attacked_atom_idx].coord
        diff_vec /= np.linalg.norm(diff_vec)
        nucleophile_coords = substrate.atoms[indices.attacked_atom_idx].coord + distance * diff_vec
        
        # generate constraints + bond_rearrangements
        constraints = {
            (h_idx, len(substrate.atoms)): distance
        }
        tmp_indices = copy(indices)
        tmp_indices.nucleophile_idx = len(substrate.atoms)
        self.indices = tmp_indices
        self.h_idx = h_idx
        bond_rearrangement = self._construct_bond_rearrangement(h_idx, tmp_indices)

        # generate reaction complex
        reaction_complex = substrate.copy()
        new_atoms = substrate.atoms + [ade.Atom(
                atomic_symbol=nucleophile.atoms[0].atomic_symbol,
                x=nucleophile_coords[0],
                y=nucleophile_coords[1],
                z=nucleophile_coords[2]
            )]
        reaction_complex._parent_atoms = [atom for atom in new_atoms]
        reaction_complex._coordinates = Coordinates(np.array([atom.coord for atom in new_atoms]))
        reaction_complex.charge = -1

        # constrained optimization of reaction complex
        reaction_complex.constraints.distance = constraints
        reaction_complex.optimise(method=method)

        # relaxed optimization of reaction complex
        reaction_complex.constraints.distance = {}
        reaction_complex.optimise(method=method)

        # check conditions
        conditions_passed = []
        for check in self.checks:
            conditions_passed.append(
                check(
                    reaction_complex,
                    h_idx,
                    indices,
                    self.dist_dict,
                    nucleophile.atoms[0].atomic_symbol
                )
            )
        
        return (False not in conditions_passed), reaction_complex, bond_rearrangement