"""
Compute dihedral angles (phi/psi) for Ac-Ala3-NHMe MD simulations
and save them as .npz files alongside the HDF5 log files.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import binned_statistic_2d

import h5py
from schnetpack.md.data import HDF5Loader
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis.dihedrals import Dihedral
import MDAnalysis as mda

from imlff.common.util import proj_path
from imlff.data.info import md17_conformations_per_molecule, md22_conformations_per_molecule


def get_molecule_name(log_path):
    for mol in set(md17_conformations_per_molecule).union(set(md22_conformations_per_molecule)):
        if f'/{mol}_' in log_path or f'/{mol}/' in log_path:
            return mol
    raise ValueError(f"Could not determine molecule name from: {log_path}")


def compute_and_save_angles(log_path):
    hdf5_path = f"{log_path}.hdf5"
    out_path = f"{log_path}_angles.npz"

    if not os.path.exists(hdf5_path):
        print(f"  SKIP (no HDF5): {hdf5_path}")
        return False

    if os.path.exists(out_path):
        print(f"  SKIP (already exists): {out_path}")
        return False

    with HDF5Loader(hdf5_path, skip_initial=0) as data:
        pass  # once exiting the context 'data' is not writable but readable

    atomic_numbers = data.get_property('_atomic_numbers', atomistic=True)

    # Build MDAnalysis Universe
    u = mda.Universe.empty(
        len(atomic_numbers),
        n_residues=len(atomic_numbers),
        atom_resindex=np.arange(len(atomic_numbers)),
        trajectory=True,
    )
    u.add_TopologyAttr('name', [f'{Z}' for Z in atomic_numbers])
    u.add_TopologyAttr('type', ['X'] * len(atomic_numbers))
    u.add_TopologyAttr('resid', list(range(1, len(atomic_numbers) + 1)))
    u.add_TopologyAttr('resname', ['MOL'] * len(atomic_numbers))
    u.add_TopologyAttr('mass', atomic_numbers)  # placeholder

    positions = data.get_property('_positions', atomistic=True)
    u.trajectory = MemoryReader(positions)

    # Ac-Ala3-NHMe dihedral atom groups
    phi_group = mda.AtomGroup([11, 16, 18, 21], u)
    psi_group = mda.AtomGroup([26, 21, 18, 16], u)

    phi = Dihedral([phi_group]).run()
    psi = Dihedral([psi_group]).run()

    phi_angles = phi.angles.reshape(-1)
    psi_angles = psi.angles.reshape(-1)

    np.savez_compressed(out_path, phi=phi_angles, psi=psi_angles)
    print(f"  SAVED: {out_path}  ({len(phi_angles)} frames)")
    return True


def save_ramachandran(phi_angles, psi_angles, save_path, bin_sizes=(1, 2, 5, 10)):
    for bs in bin_sizes:
        n_bins = 360 // bs
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist2d(phi_angles, psi_angles, bins=n_bins, cmap=CMAPS['Blues'], range=[[-180, 180], [-180, 180]])
        ax.set_xlabel('Phi')
        ax.set_ylabel('Psi')
        ax.set_title(f'Ramachandran ({bs} deg bins)')
        plt.colorbar(ax.collections[0], ax=ax)
        plt.tight_layout()
        out = f"{save_path}_ramachandran_{bs}deg.pdf"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  PLOT: {out}")


def _plot_binned_properties(phi_angles, psi_angles, properties, save_path, prefix,
                            bin_sizes=(1, 2, 5, 10), energy=None, top_k_energy=None):
    """Plot min/mean of properties binned by phi/psi angles.

    properties: dict of {name: 1d array}
    prefix: string prepended to output filenames (e.g. '' or 'rerun_cold_')
    energy: optional 1d array of energies (same length as phi/psi). If provided
        together with top_k_energy, only the top_k_energy lowest-energy frames
        in each bin are used to compute min/mean.
    top_k_energy: int or None. If None, all frames in each bin are used.
    """
    ang_range = [[-180, 180], [-180, 180]]
    energy_suffix = f'_top{top_k_energy}e' if (top_k_energy is not None and energy is not None) else ''

    _cbar_label = {
        'forw_iterations': 'Forward Iterations',
        'back_iterations': 'Backward Iterations',
        'mean_iterations': 'Iterations',
        'energy': 'Energy',
    }

    for prop_name, values in properties.items():
        n = min(len(phi_angles), len(values))
        if energy is not None:
            n = min(n, len(energy))
        phi, psi, vals = phi_angles[:n], psi_angles[:n], values[:n]
        nrg = energy[:n] if energy is not None else None

        cbar_label = _cbar_label.get(prop_name, prop_name)
        # title label: drop _iterations suffix, keep other parts of prefix
        title_prop = prop_name.replace('_iterations', '').replace('_', ' ')

        for bs in bin_sizes:
            n_bins = 360 // bs

            if top_k_energy is not None and nrg is not None:
                _, x_edges, y_edges, binnumber = binned_statistic_2d(
                    phi, psi, vals, statistic='mean',
                    bins=n_bins, range=ang_range,
                )
                nx, ny = n_bins, n_bins
                stat_min = np.full((nx, ny), np.nan)
                stat_mean = np.full((nx, ny), np.nan)
                for xi in range(nx):
                    for yi in range(ny):
                        flat_idx = (xi + 1) * (n_bins + 2) + (yi + 1)
                        mask = binnumber == flat_idx
                        if not mask.any():
                            continue
                        bin_e = nrg[mask]
                        bin_v = vals[mask]
                        k = min(top_k_energy, len(bin_e))
                        top_idx = np.argpartition(bin_e, k - 1)[:k]
                        top_v = bin_v[top_idx]
                        stat_min[xi, yi] = top_v.min()
                        stat_mean[xi, yi] = top_v.mean()
                title_extra = f', top{top_k_energy} E'
            else:
                stat_min, x_edges, y_edges, _ = binned_statistic_2d(
                    phi, psi, vals, statistic='min',
                    bins=n_bins, range=ang_range,
                )
                stat_mean, _, _, _ = binned_statistic_2d(
                    phi, psi, vals, statistic='mean',
                    bins=n_bins, range=ang_range,
                )
                title_extra = ''

            vmin = np.nanmin([np.nanmin(stat_min), np.nanmin(stat_mean)])
            vmax = np.nanmax([np.nanmax(stat_min), np.nanmax(stat_mean)])

            for cmap_name, cmap in CMAPS.items():
                fig, axes = plt.subplots(1, 2, figsize=(13, 5))
                for ax, stat_name, stat in zip(axes, ['min', 'mean'], [stat_min, stat_mean]):
                    im = ax.imshow(
                        stat.T, origin='lower',
                        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                        cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto',
                    )
                    plt.colorbar(im, ax=ax, label=cbar_label)
                    ax.set_xlabel(r'$\phi$')
                    ax.set_ylabel(r'$\psi$')
                    ax.set_title(f'{stat_name} {title_prop} ({bs}°{title_extra})')

                plt.tight_layout()
                out = f"{save_path}_{prefix}{prop_name}{energy_suffix}_{bs}deg_cmap{cmap_name}.pdf"
                fig.savefig(out, dpi=150)
                plt.close(fig)
                print(f"  PLOT: {out}")


def save_binned_property_plots(phi_angles, psi_angles, data, save_path,
                               bin_sizes=(10,), top_k_energy=None):
    properties = {}
    if 'forw_iterations' in data.properties:
        properties['forw_iterations'] = data.properties['forw_iterations'].flatten()
    if 'back_iterations' in data.properties:
        properties['back_iterations'] = data.properties['back_iterations'].flatten()

    energy = data.properties['energy'].flatten() if 'energy' in data.properties else None
    if energy is not None:
        properties['energy'] = energy

    if 'forw_iterations' in properties and 'back_iterations' in properties:
        properties['mean_iterations'] = (properties['forw_iterations'] + properties['back_iterations']) / 2

    _plot_binned_properties(phi_angles, psi_angles, properties, save_path, prefix='',
                            bin_sizes=bin_sizes, energy=energy, top_k_energy=top_k_energy)


def _iters_to_converge(res, threshold):
    """Given residuals (n_steps, n_atoms, max_iter), find first iter where ALL atoms < threshold.

    Returns (n_steps,) float array. If never converged, returns max_iter.
    """
    n_steps, n_atoms, max_iter = res.shape
    all_below = np.all(np.where(np.isnan(res), np.inf, res) < threshold, axis=1)  # (n_steps, max_iter)
    has_any = all_below.any(axis=1)
    first_converged = np.where(has_any, np.argmax(all_below, axis=1), max_iter)
    return first_converged.astype(float)


def _iters_cache_path(rerun_path, direction, n_rerun, threshold):
    """Path for cached iteration counts: <rerun_path>_<direction>_iters_n{n_rerun}_tol{threshold}.npy"""
    return f"{rerun_path}_{direction}_iters_n{n_rerun}_tol{threshold}.npy"


def get_n_rerun(rerun_path):
    """Get n_rerun from the h5 file without loading data into memory."""
    try:
        f = h5py.File(rerun_path, 'r', swmr=True)
    except Exception:
        f = h5py.File(rerun_path, 'r')
    with f:
        return f['forw_res'].shape[0]


def precompute_rerun_iters(rerun_path, forw_threshold=0.01, back_threshold=None, n_rerun=None):
    """Load forw_res/back_res one at a time, compute iteration counts, save as int16 .npy files.

    Saves to: <rerun_path>_{forw,back}_iters_n{n_rerun}_tol{threshold}.npy
    Skips if files already exist. Returns (forw_cache_path, back_cache_path, n_rerun).
    n_rerun: if given, use this value to locate existing cache files instead of reading the HDF5.
    """
    if back_threshold is None:
        back_threshold = forw_threshold

    if n_rerun is None:
        n_rerun = get_n_rerun(rerun_path)

    forw_cache = _iters_cache_path(rerun_path, 'forw', n_rerun, forw_threshold)
    back_cache = _iters_cache_path(rerun_path, 'back', n_rerun, back_threshold)

    if not os.path.exists(forw_cache):
        print(f"  Computing forw_iters (n_rerun={n_rerun}, tol={forw_threshold})...")
        try:
            f = h5py.File(rerun_path, 'r', swmr=True)
        except Exception:
            f = h5py.File(rerun_path, 'r')
        with f:
            forw_res = f['forw_res'][:n_rerun]
        forw_iters = _iters_to_converge(forw_res, forw_threshold).astype(np.int16)
        del forw_res
        np.save(forw_cache, forw_iters)
        print(f"  SAVED: {forw_cache}")
    else:
        print(f"  SKIP (exists): {forw_cache}")

    if not os.path.exists(back_cache):
        print(f"  Computing back_iters (n_rerun={n_rerun}, tol={back_threshold})...")
        try:
            f = h5py.File(rerun_path, 'r', swmr=True)
        except Exception:
            f = h5py.File(rerun_path, 'r')
        with f:
            back_res = f['back_res'][:n_rerun]
        back_iters = _iters_to_converge(back_res, back_threshold).astype(np.int16)
        del back_res
        np.save(back_cache, back_iters)
        print(f"  SAVED: {back_cache}")
    else:
        print(f"  SKIP (exists): {back_cache}")

    return forw_cache, back_cache, n_rerun


def _rerun_prefix(every=1, max_iter=10):
    """Build prefix string matching rerun filename markers."""
    parts = ['rerun_cold']
    if every != 1:
        parts.append(f'every{every}')
    if max_iter != 10:
        parts.append(f'mi{max_iter}')
    return '_'.join(parts)


def _rerun_h5_path(log_path, every=1, max_iter=10):
    """Build rerun .h5 path matching rerun_md_traj._rerun_filename."""
    name = f"{log_path}_rerun"
    if every != 1:
        name += f"_every{every}"
    if max_iter != 10:
        name += f"_mi{max_iter}"
    return name + ".h5"


def save_rerun_binned_plots(phi_angles, psi_angles, rerun_path, save_path,
                            forw_threshold=0.01, back_threshold=None,
                            every=1, max_iter=10,
                            bin_sizes=(1, 2, 5, 10),
                            energy=None, top_k_energy=None,
                            n_rerun=None):
    """Plot binned rerun iteration counts. Uses precomputed int16 caches (runs precompute if needed)."""
    if back_threshold is None:
        back_threshold = forw_threshold

    forw_cache, back_cache, n_rerun = precompute_rerun_iters(
        rerun_path, forw_threshold, back_threshold, n_rerun=n_rerun,
    )

    forw_iters = np.load(forw_cache).astype(float)
    back_iters = np.load(back_cache).astype(float)

    # Subsample angles to match the rerun's every-N spacing
    if every > 1:
        phi = phi_angles[::every][:n_rerun]
        psi = psi_angles[::every][:n_rerun]
    else:
        phi = phi_angles[:n_rerun]
        psi = psi_angles[:n_rerun]

    base_prefix = _rerun_prefix(every=every, max_iter=max_iter)
    ftol = f'tol{forw_threshold}'
    btol = f'tol{back_threshold}'

    plot_kw = dict(bin_sizes=bin_sizes, energy=energy, top_k_energy=top_k_energy)

    # forw: only forw_threshold in name
    _plot_binned_properties(
        phi, psi, {'forw_iterations': forw_iters},
        save_path, prefix=f'{base_prefix}_{ftol}_', **plot_kw,
    )
    # back: only back_threshold in name
    _plot_binned_properties(
        phi, psi, {'back_iterations': back_iters},
        save_path, prefix=f'{base_prefix}_{btol}_', **plot_kw,
    )
    # mean: crop to shorter if lengths differ, note in filename
    n_mean = min(len(forw_iters), len(back_iters))
    if forw_threshold == back_threshold:
        mtol = f'tol{forw_threshold}'
    else:
        mtol = f'tol{forw_threshold}_btol{back_threshold}'
    mean_suffix = f'_n{n_mean}' if len(forw_iters) != len(back_iters) else ''
    phi_m, psi_m = phi[:n_mean], psi[:n_mean]
    nrg_m = energy[:n_mean] if energy is not None else None
    _plot_binned_properties(
        phi_m, psi_m,
        {'mean_iterations': (forw_iters[:n_mean] + back_iters[:n_mean]) / 2},
        save_path, prefix=f'{base_prefix}_{mtol}{mean_suffix}_',
        bin_sizes=bin_sizes, energy=nrg_m, top_k_energy=top_k_energy,
    )
    
  
    
def _truncate_cmap(name, low=0.05, high=1.0, n=256):
    return LinearSegmentedColormap.from_list(
        f'{name}_t', plt.get_cmap(name)(np.linspace(low, high, n))
    )


CMAPS = {name: _truncate_cmap(name) for name in ('Blues', 'Greens', 'Purples')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('array_idx', type=int, nargs='?', default=None)
    parser.add_argument('--no-plot', action='store_false', dest='plot', help='Disable Ramachandran plots')
    parser.add_argument('--every', type=int, default=2, help='Rerun step spacing to plot (default: 2)')
    parser.add_argument('--max-iter', type=int, default=20, help='Max iter used in rerun (default: 20)')
    parser.add_argument('--run-id', type=str, default=None, help='Filter by run_id')
    parser.add_argument('--top-k-energy', type=int, default=None, help='Per bin: use only the N lowest-energy frames (default: all)')
    parser.add_argument('--n-rerun', type=int, default=1000000, help='Number of frames in rerun cache to use (default: 1000000)')
    parser.add_argument('--warm', action=argparse.BooleanOptionalAction, default=True, help='Plot warm (in-MD) iteration counts')
    parser.add_argument('--cold', action=argparse.BooleanOptionalAction, default=True, help='Plot cold-rerun iteration counts')
    args = parser.parse_args()

    csv_path = proj_path('result_summary/feb-16/md-Ala3.csv')
    sims_df = pd.read_csv(csv_path).sort_values('_step', ascending=False)
    if args.run_id is not None:
        sims_df = sims_df.query(f'run_id=="{args.run_id}"')

    print(f"Loaded {len(sims_df)} rows from {csv_path}")

    for i, row in sims_df.iterrows():
        if args.array_idx is not None and i != args.array_idx:
            continue

        pat = row.proj_path
        if not type(pat) is str:
            continue
        log_path = proj_path(pat)
        molecule_name = get_molecule_name(log_path)

        parts = log_path.split(molecule_name, 1)[1].split('/', 1)
        if len(parts) > 1:
            plot_dir = parts[1]
            plot_path = proj_path(f'plots/md/{molecule_name}/{plot_dir}')
            os.makedirs(plot_path, exist_ok=True)

        print(f"[{i}/{len(sims_df)}] {row.proj_path}")
        compute_and_save_angles(log_path)

        if args.plot:
            angle_path = f"{log_path}_angles.npz"
            hdf5_path = f"{log_path}.hdf5"
            rerun_path = _rerun_h5_path(log_path, every=args.every, max_iter=args.max_iter)

            if not os.path.exists(angle_path):
                continue
            
            energy = None
            if os.path.exists(hdf5_path):
                dat = np.load(angle_path)
                with HDF5Loader(hdf5_path, skip_initial=0) as hdf5_data:
                    pass
                if args.warm:
                    save_binned_property_plots(
                        dat['phi'], dat['psi'], hdf5_data, log_path,
                        top_k_energy=args.top_k_energy,
                    )
                if 'energy' in hdf5_data.properties:
                    energy = hdf5_data.properties['energy'].flatten()

            if os.path.exists(rerun_path) and args.cold:
                save_rerun_binned_plots(
                    dat['phi'], dat['psi'], rerun_path, log_path,
                    every=args.every, max_iter=args.max_iter,
                    energy=energy,
                    top_k_energy=args.top_k_energy,
                    n_rerun=args.n_rerun,
                )


if __name__ == '__main__':
    main()