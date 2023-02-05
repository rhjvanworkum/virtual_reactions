


from typing import List, Tuple, Union
import numpy as np

import scipy
from scipy import spatial

from utils import Atom


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


class ReactionTemplate:

    def __init__(self) -> None:
        pass

    def _generate_ts_geometry(
        self,
        substrate_geometry: List[Atom],
        nucleophile: str,
    ) -> List[Atom]:
        nucleophile = Atom(
            type=''.join([c for c in nucleophile if c.isupper()]),
            x=0.0,
            y=0.0,
            z=0.0
        )

        for idx, atom in enumerate(substrate_geometry):
            if atom.type == 'H':
                insert_idx = idx
                break

        ts_geometry = substrate_geometry
        ts_geometry.insert(insert_idx, nucleophile)

        return ts_geometry

    def generate_ts(
        self,
        substrate_geometry: List[Atom],
        nucleophile: str,
        indices: E2Sn2ReactionIndices
    ) -> Tuple[List[Atom], List[Tuple[Tuple[int], float]]]:
        raise NotImplementedError



class Sn2ReactionTemplate(ReactionTemplate):

    def __init__(
        self,
        d_nucleophile: float,
        d_leaving_group: float,
        angle: float
    ) -> None:
        super(Sn2ReactionTemplate, self).__init__()
        self.d_nucleophile = d_nucleophile
        self.d_leaving_group = d_leaving_group
        self.angle = angle

    def generate_ts(
        self,
        substrate_geometry: List[Atom],
        nucleophile: str,
        indices: E2Sn2ReactionIndices,
    ) -> List[Atom]:
        ts_geometry = self._generate_ts_geometry(substrate_geometry, nucleophile)

        # set distance and angle
        # 1. scale distance vector of leaving group
        diff_vec = ts_geometry[indices.leaving_group_idx].coordinates - ts_geometry[indices.central_atom_idx].coordinates
        diff_vec /= np.linalg.norm(diff_vec)
        ts_geometry[indices.leaving_group_idx].coordinates = ts_geometry[indices.central_atom_idx].coordinates + self.d_leaving_group * diff_vec

        # 2. set nucleophile vector to rotated (and scaled) version of leaving group
        rot_vec = ts_geometry[indices.central_atom_idx].coordinates - ts_geometry[indices.attacked_atom_idx].coordinates
        rot_vec = rot_vec / np.linalg.norm(rot_vec) * np.radians(self.angle)
        r = spatial.transform.Rotation.from_rotvec(rot_vec)
        ts_geometry[indices.nucleophile_idx].coordinates = ts_geometry[indices.central_atom_idx].coordinates + r.apply(self.d_nucleophile * diff_vec)

        constraints = [
            ((indices.central_atom_idx, indices.leaving_group_idx), self.d_leaving_group),
            ((indices.central_atom_idx, indices.nucleophile_idx), self.d_nucleophile)
        ]

        return ts_geometry, constraints


def get_hs(ts_geometry, indices):
    total_hs = []

    for idx, atom in enumerate(ts_geometry):
        if atom.type == 'H':
            total_hs.append((idx, atom))

    hs = []
    for (idx, atom) in total_hs:
        if np.linalg.norm(ts_geometry[indices.attacked_atom_idx].coordinates - atom.coordinates) < 1.2:
            hs.append(idx)

    if len(hs) > 3:
        print('C has more than 3 hydrogen bonded to them!')
        raise ValueError("notallowedC")

    return hs

def get_opposite_hydrogen(
    ts_geometry,
    indices,
    hs_indices
) -> int:
    leaving_group_vec = ts_geometry[indices.leaving_group_idx].coordinates - ts_geometry[indices.central_atom_idx].coordinates
    leaving_group_vec /= np.linalg.norm(leaving_group_vec)

    angles = []
    for h_idx in hs_indices:
        vec = ts_geometry[h_idx].coordinates - ts_geometry[indices.attacked_atom_idx].coordinates
        vec /= np.linalg.norm(vec)
        angle = np.arccos(np.dot(leaving_group_vec, vec))
        angles.append(angle)
    
    return hs_indices[np.argmin(np.array(angles))]




class E2ReactionTemplate(ReactionTemplate):

    """ Assume for now this is always TRANS """

    def __init__(
        self,
        d_nucleophile: float,
        d_H: float,
        d_leaving_group: float,
    ) -> None:
        super(E2ReactionTemplate, self).__init__()
        self.d_nucleophile = d_nucleophile
        self.d_H = d_H
        self.d_leaving_group = d_leaving_group

    def generate_ts(
        self,
        substrate_geometry: List[Atom],
        nucleophile: str,
        indices: E2Sn2ReactionIndices,
    ) -> List[Atom]:
        ts_geometry = self._generate_ts_geometry(substrate_geometry, nucleophile)

        # get close by H's
        hs_indices = get_hs(ts_geometry, indices)
        h_idx = get_opposite_hydrogen(ts_geometry, indices, hs_indices)

        # 1. scale distance vector of leaving group
        diff_vec = ts_geometry[indices.leaving_group_idx].coordinates - ts_geometry[indices.central_atom_idx].coordinates
        diff_vec /= np.linalg.norm(diff_vec)
        ts_geometry[indices.leaving_group_idx].coordinates = ts_geometry[indices.central_atom_idx].coordinates + self.d_leaving_group * diff_vec

        # 2. set hydrogen distance
        diff_vec = ts_geometry[h_idx].coordinates - ts_geometry[indices.attacked_atom_idx].coordinates
        diff_vec /= np.linalg.norm(diff_vec)
        ts_geometry[h_idx].coordinates = ts_geometry[indices.attacked_atom_idx].coordinates + self.d_H * diff_vec

        # 3. set nucleohpile distance
        ts_geometry[indices.nucleophile_idx].coordinates = ts_geometry[indices.attacked_atom_idx].coordinates + (self.d_H + self.d_nucleophile) * diff_vec

        constraints = [
            ((indices.central_atom_idx, indices.leaving_group_idx), self.d_leaving_group),
            ((indices.attacked_atom_idx, h_idx), self.d_H),
            ((h_idx, indices.nucleophile_idx), self.d_nucleophile)
        ]

        return ts_geometry, constraints
