"""
Re-run an MD trajectory through the DEQ model to record per-step
forward/backward iteration counts and residuals.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import h5py

from schnetpack.md.data import HDF5Loader

from imlff.common.util import proj_path
from imlff.loading import load_model_from_wandb
from imlff.model.platform import wait_for_cuda
from imlff.analysis.molecules import create_batch
from imlff.scripts.compute_dihedral_angles import get_molecule_name
from tqdm import tqdm

BATCH_SIZE = 5
DEFAULT_CUTOFF = 5.0
FLUSH_EVERY = 1000  # flush to HDF5 every this many frames


def _pad_and_concat(res_list, n_atoms):
    """Pad residual arrays to a common max_iter dimension and concatenate."""
    max_k = max(a.shape[2] for a in res_list)
    padded = []
    for a in res_list:
        if a.shape[2] < max_k:
            pad = np.full((a.shape[0], n_atoms, max_k - a.shape[2]), np.nan, dtype=np.float32)
            a = np.concatenate([a, pad], axis=2)
        padded.append(a)
    return np.concatenate(padded, axis=0)


def rerun_trajectory(nnpot, positions, Z, device, out_path, cutoff=DEFAULT_CUTOFF,
                     batch_size=BATCH_SIZE, flush_every=FLUSH_EVERY):
    """Run positions through the model in batches, streaming results to HDF5.
    """

    n_steps = positions.shape[0]
    n_atoms = positions.shape[1]
    init_max_iter = nnpot.representation.deq.forw_max_iter

    # Determine resume point from existing file
    start_frame = 0
    if os.path.exists(out_path):
        with h5py.File(out_path, 'r') as f:
            if 'forw_res' in f:
                start_frame = f['forw_res'].shape[0]
        if start_frame >= n_steps:
            print(f"  Already complete ({start_frame}/{n_steps})")
            return
        print(f"  Resuming from frame {start_frame}/{n_steps}")

    # Use libver='latest' for new files (enables SWMR), plain 'a' for existing ones
    is_new = not os.path.exists(out_path) or start_frame == 0
    open_kwargs = dict(libver='latest') if is_new else {}

    with h5py.File(out_path, 'a', **open_kwargs) as f:
        # Create or open resizable datasets (iteration dim is unlimited)
        if 'forw_res' not in f:
            f.create_dataset('forw_res', shape=(0, n_atoms, init_max_iter), maxshape=(n_steps, n_atoms, None), dtype='f4')
            f.create_dataset('back_res', shape=(0, n_atoms, init_max_iter), maxshape=(n_steps, n_atoms, None), dtype='f4')

        try:
            f.swmr_mode = True  # allow concurrent readers
        except ValueError:
            pass  # file was created without libver='latest', SWMR not supported

        ds_fr = f['forw_res']
        ds_br = f['back_res']

        # Accumulators for the current flush window
        buf_fr, buf_br = [], []
        frames_since_flush = 0

        def flush():
            nonlocal buf_fr, buf_br, frames_since_flush
            if not buf_fr:
                return
            fr = _pad_and_concat(buf_fr, n_atoms)
            br = _pad_and_concat(buf_br, n_atoms) if buf_br else np.full_like(fr, np.nan)

            n = fr.shape[0]
            old = ds_fr.shape[0]
            for ds, arr in [(ds_fr, fr), (ds_br, br)]:
                new_shape = list(ds.shape)
                new_shape[0] = old + n
                ds.resize(new_shape)
                ds[old:old + n] = arr

            f.flush()
            buf_fr, buf_br = [], []
            frames_since_flush = 0

        for start in tqdm(range(start_frame, n_steps, batch_size),
                          initial=start_frame // batch_size,
                          total=(n_steps + batch_size - 1) // batch_size):
            end = min(start + batch_size, n_steps)
            pos_batch = torch.tensor(positions[start:end], dtype=torch.float32)
            bs = pos_batch.shape[0]
            pos_flat = pos_batch.reshape(bs * n_atoms, 3)

            Z_batch = Z.repeat(bs)
            idx_m = torch.arange(bs).repeat_interleave(n_atoms)

            batch = create_batch(pos_flat, Z=Z_batch, connectivity=cutoff, device=device, idx_m=idx_m)

            out = nnpot(batch)
            del batch  # free GPU memory immediately

            for buf, key in [(buf_fr, 'forw_res'), (buf_br, 'back_res')]:
                res_dict = out[key]
                if res_dict:
                    arr = np.full((bs, n_atoms, init_max_iter), np.nan, dtype=np.float32)
                    for k, v in res_dict.items():
                        if k >= init_max_iter:
                            continue
                        arr[:, :, k] = v.detach().cpu().numpy().reshape(bs, n_atoms)
                    buf.append(arr)

            del out  # free references to model output
            frames_since_flush += bs
            if frames_since_flush >= flush_every:
                flush()

        # Final flush
        flush()


def _rerun_filename(log_file, every=1, max_iter=10):
    """Build rerun output filename with markers for non-default settings."""
    name = f"{log_file}_rerun"
    if every != 1:
        name += f"_every{every}"
    if max_iter != 10:
        name += f"_mi{max_iter}"
    return name + ".h5"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('array_idx', type=int, nargs='?', default=None)
    parser.add_argument('--every', type=int, default=1, help='Evaluate every N-th MD step (default: 1)')
    parser.add_argument('--max-iter', type=int, default=10, help='Max forward/backward iterations (default: 10)')
    parser.add_argument('--run-id', type=str, default=None, help='Filter by run_id')
    args = parser.parse_args()

    csv_path = proj_path('result_summary/feb-16/md-Ala3.csv')
    sims_df = pd.read_csv(csv_path).sort_values('_step', ascending=False)
    if args.run_id is not None and args.run_id:
        sims_df = sims_df.query(f'run_id=="{args.run_id}"')

    print(f"Loaded {len(sims_df)} rows from {csv_path}")

    device = wait_for_cuda()

    # Cache loaded models by tag to avoid reloading
    loaded_models = {}

    for i, row in sims_df.iterrows():
        if args.array_idx is not None and i != args.array_idx:
            continue

        pat = row.proj_path
        if not type(pat) is str:
            continue
        log_file = proj_path(pat)
        molecule_name = get_molecule_name(log_file)

        parts = log_file.split(molecule_name, 1)[1].split('/', 1)
        if len(parts) > 1:
            plot_dir = parts[1]
            plot_path = proj_path(f'plots/md/{molecule_name}/{plot_dir}')
            os.makedirs(plot_path, exist_ok=True)

        hdf5_path = f"{log_file}.hdf5"
        if not os.path.exists(hdf5_path):
            print(f"[{i}/{len(sims_df)}] SKIP (no hdf5): {row.proj_path}")
            continue

        out_path = _rerun_filename(log_file, every=args.every, max_iter=args.max_iter)

        print(f"[{i}/{len(sims_df)}] {row.proj_path}")
        print(f"  -> {out_path}")

        # Load trajectory data, extract positions and free HDF5Loader immediately
        with HDF5Loader(hdf5_path, skip_initial=0) as data:
            pass

        Z = torch.tensor(data.properties['_atomic_numbers'], dtype=torch.long)
        positions = data.properties['_positions'][:, 0][::args.every]  # (n_steps, n_atoms, 3)
        del data  # free memory

        # Load or retrieve cached model
        model_tag = row.model_config__model_tag
        if model_tag not in loaded_models:
            nnpot = load_model_from_wandb(model_tag).to(device)
            nnpot.representation.deq.implicit_diff = True
            nnpot.representation.deq.forw_tol = 1e-4
            nnpot.representation.deq.back_tol = 1e-4
            nnpot.representation.deq.forw_max_iter = args.max_iter
            nnpot.representation.deq.back_max_iter = args.max_iter
            nnpot.representation.deq.back_iter_on_g_bar = False # TODO: True
            nnpot.eval()
            loaded_models[model_tag] = nnpot
        nnpot = loaded_models[model_tag]

        # Run trajectory through model, streaming to HDF5
        cutoff = row.train__cutoff if pd.notna(row.train__cutoff) else DEFAULT_CUTOFF
        rerun_trajectory(nnpot, positions, Z, device, out_path, cutoff=cutoff)
        del positions # free memory
        print(f"  Done: {out_path}")



if __name__ == '__main__':
    main()
