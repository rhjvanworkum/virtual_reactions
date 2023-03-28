import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import h5py

import autode as ade
from autode.opt.optimisers import PRFOptimiser
from autode.input_output import xyz_file_to_atoms
from autode.wrappers.XTB import XTB
from src.reactions.e2_sn2.old.template import E2Sn2ReactionIndices

from a_export_sn2_structures import sn2_reaction_complex_template
from src.utils import write_xyz_file

from vis_normal_modes import plot_normal_modes


if __name__ == "__main__":
    ade.Config.n_cores = 4
    ade.Config.XTB.path = '/home/ruard/Programs/xtb-6.5.1/bin/xtb'

    rc = ade.Species(
        name='rc',
        atoms=xyz_file_to_atoms('/home/ruard/code/virtual_reactions/tempt/rc2.xyz'),
        charge=-1,
        mult=1
    )

    pc = ade.Species(
        name='pc',
        atoms=xyz_file_to_atoms('/home/ruard/code/virtual_reactions/tempt/pc2.xyz'),
        charge=-1,
        mult=1
    )

    method = XTB()

    ts_optimizer = PRFOptimiser(
        maxiter=100,
        gtol=1e-3,
        etol=1e-3,
        init_alpha=0.01,
        recalc_hessian_every=3
    )

    from geodesic_interpolate.fileio import read_xyz, write_xyz
    from geodesic_interpolate.interpolation import redistribute
    from geodesic_interpolate.geodesic import Geodesic

    nimages = 20
    tol = 1e-2
    scaling = 1.7
    dist_cutoff = 3
    friction = 1e-2
    maxiter = 15


    symbols = [atom.atomic_symbol for atom in rc.atoms]
    X = [rc.coordinates, pc.coordinates]
    raw = redistribute(symbols, X, nimages, tol=tol * 5)
    smoother = Geodesic(symbols, raw, scaling, threshold=dist_cutoff, friction=friction)
    try:
        smoother.smooth(tol=tol, max_iter=maxiter)
    finally:
        # TODO: write path
        # write_xyz('test.xyz', symbols, smoother.path)

        energies = []
        complexes = []
        for idx, geom in enumerate(smoother.path):
            complex = ade.Species(
                name=f'test_complex_{idx}',
                atoms=rc.atoms,
                charge=-1,
                mult=1
            )
            complex.coordinates = geom
            complexes.append(complex)

            complex.single_point(method=method)
            energies.append(complex.energy)

            complex.calc_hessian(method=method)

            # -894.2481807099042
            # -1029.9285391635367

            for idx, freq in enumerate(complex.hessian.frequencies_proj):
                if freq.is_imaginary: # and (freq > -1030 and freq < -1029): # (freq > -895 and freq < -894):
                    plot_normal_modes(
                        atom_symbols=[atom.atomic_symbol for atom in complex.atoms],
                        R=complex.coordinates,
                        normal_modes=complex.hessian.normal_modes_proj[idx]
                    )
                    ts_guess = complex
                    break


            # for idx, freq in enumerate(complex.hessian.frequencies_proj):
            #     if freq.is_imaginary:
            #         print(freq)
            #         plot_normal_modes(
            #             atom_symbols=[atom.atomic_symbol for atom in complex.atoms],
            #             R=complex.coordinates,
            #             normal_modes=complex.hessian.normal_modes_proj[idx]
            #         )


            # print(complex.imaginary_frequencies)
            # if complex.imaginary_frequencies is not None:
            #     print(complex.frequencies)
            #     # print(complex.frequencies[:len(complex.imaginary_frequencies)])
            # print('\n')

        # TODO: save energy of path
        # plt.plot(np.arange(len(energies)), energies)
        # plt.show()
        # print(np.argmax(np.array(energies)))

        # matched = False
        # idx = 0
        # while not matched:
        #     ts_guess = complexes[nimages // 2 + idx]
        #     ts_guess.calc_hessian(method=method)
        #     if ts_guess.imaginary_frequencies is not None:
        #         if len(ts_guess.imaginary_frequencies) == 1:
        #             matched = True
        #     idx += 1

    print(ts_guess.imaginary_frequencies)

    # # # # ts_guess = rc.copy()
    # # # # ts_guess.coordinates = rc.coordinates + 0.5 * (pc.coordinates - rc.coordinates)
    # write_xyz_file(ts_guess.atoms, 'tempt/ts_guess2.xyz')

    ts_optimizer.optimise(
        species=ts_guess,
        method=method,
        maxiter=100
    )
    ts_guess.calc_hessian(method=method)
    print(f'imag frequency of optimized TS: {ts_guess.imaginary_frequencies}')
    write_xyz_file(ts_guess.atoms, 'ts_opt.xyz')

    # # print(ts_guess.hessian.frequencies)
    # # print(ts_guess.hessian.normal_modes
    # # 
    # print(ts_guess.hessian.frequencies[0])
    # # print(ts_guess.hessian.normal_modes[0])
    # print(ts_guess.hessian.normal_modes_proj[0].shape)

    for i in range(len(ts_guess.hessian.frequencies_proj)):
        if ts_guess.hessian.frequencies_proj[i].is_imaginary:
            normal_mode = ts_guess.hessian.normal_modes_proj[i]
            break
    #         print(ts_guess.hessian.frequencies_proj[i])
    #         # plot_normal_modes(
    #         #     atom_symbols=[atom.atomic_symbol for atom in ts_guess.atoms],
    #         #     R=ts_guess.coordinates,
    #         #     normal_modes=ts_guess.hessian.normal_modes_proj[i]
    #         # )

    # # print(ts_guess.hessian)
    # with h5py.File('ts_opt_hessian.h5', "w") as handle:
    #     handle.create_dataset("hessian", data=ts_guess.hessian)

    # # """ IRC """
    from autode.path.irc import IRC

    ts = ade.Species(
        name='ts',
        atoms=ts_guess.atoms,
        charge=-1,
        mult=1
    )

    irc = IRC()
    ts = irc.run(
        species=ts,
        method=method,
        initial_direction=normal_mode
    )
    write_xyz_file(ts.atoms, 'ts_steepest_descent.xyz')
    