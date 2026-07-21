import argparse
import gc
import math
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from tqdm import tqdm


class _DEQGCCallback(pl.Callback):
    """GC + cache clear between test batches for large DEQ models (e.g. DWNT)."""
    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        gc.collect()
        torch.cuda.empty_cache()

from imlff.common.util import proj_path
from imlff.data.info import molecule_to_atomic_numbers
from imlff.data.util import load_md17_data
from imlff.loading import load_model_from_wandb, artifact_or_run_cached
from imlff.scripts.util import cleanup_gpu, run_with_oom_retry


_DATA_LOAD_KEYS = [
    "num_train", "num_val", "molecule", "split_id",
    "adj_cutoff",
    "duplicate_for_run",
]


@lru_cache(maxsize=1)
def _load_data_cached(**kwargs):
    """Cache data module keyed on everything except batch_size."""
    return load_md17_data(**kwargs)


def _get_dm(data_config: dict):
    """Return cached data module with test batch_size applied."""
    batch_size = data_config.get('batch_size', 1)
    load_kwargs = {k: v for k, v in data_config.items() if k in _DATA_LOAD_KEYS}
    load_kwargs.setdefault('num_workers', 4)
    dm = _load_data_cached(**load_kwargs)
    if getattr(dm, 'test_batch_size', None) != batch_size:
        dm.test_batch_size = batch_size
        dm._test_dataloader = None  # force re-creation with updated batch_size
    return dm


# ── DEQ inference config (edit here or override as kwargs) ───────────────────
implicit_diff = True
forw_max_iter = back_max_iter = 10
forw_tol = back_tol = -1
# ─────────────────────────────────────────────────────────────────────────────


def run_test(
    tag: str | None = None,
    array_idx: int | None = None,
    array_stride: int | None = None,
    which_tag: str = 'best_manual',
    results_dir: str = 'results',
    use_tmp: bool = False,
    duplicate_for_run: bool = False,
    BSZ: int = 10_000,
    implicit_diff: bool = implicit_diff,
    forw_max_iter: int = forw_max_iter,
    back_max_iter: int = back_max_iter,
    forw_tol: float = forw_tol,
    back_tol: float = back_tol,
    max_worsening: float = 1.05,
    check_incomplete: bool = False,
    reverse: bool = False,
    batch_size: int | None = None,
    on_val: bool = False,
    csv_only: bool = False,
) -> None:
    """Run test set evaluation for a selection of models.

    If tag is given, reads runs_df_val.csv from result_summary/runs/{tag}/ and
    uses the 'best_manual_tag' column as the model. Otherwise falls back to
    result_summary/complete_test_tmp.csv with which_tag as the column prefix.
    """
    if tag is not None:
        base_name = 'runs_df_val' if on_val else 'runs_best_per_group'
        full_csv = proj_path(f'result_summary/runs/{tag}/{base_name}.csv')
        incomplete_csv = proj_path(f'result_summary/runs/{tag}/{base_name}_incomplete.csv')
        if array_idx is not None and os.path.exists(incomplete_csv):
            csv_path = incomplete_csv
        else:
            csv_path = full_csv
        df = pd.read_csv(csv_path)
        print(f"Reading {csv_path} ({len(df)} rows)")
        model_tag_col = 'best_manual_tag'
    else:
        df = pd.read_csv(proj_path('result_summary/complete_test_tmp.csv'))
        model_tag_col = f'{which_tag}_tag'

    # Sort by significant data config columns so each worker's contiguous
    # chunk shares the same data loader as much as possible.
    sort_cols = [c for c in [
        'molecule', 'num_train', 'num_val', 'split_id', 'adj_cutoff',
    ] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    val_col = f'{model_tag_col.removesuffix("_tag")}_val_forces_MAE'
    logged_col = 'val_min_logged__val_forces_MAE'
    if val_col in df.columns and logged_col in df.columns:
        before = len(df)
        both = df[val_col].notna() & df[logged_col].notna()
        within = df[val_col] <= df[logged_col] * max_worsening
        df = df[~both | within].reset_index(drop=True)
        print(f"max_worsening={max_worsening}: kept {len(df)}/{before} rows "
              f"({before - len(df)} skipped as >{max_worsening}x their logged min)")
    else:
        missing = [c for c in [val_col, logged_col] if c not in df.columns]
        print(f"Warning: max_worsening filter skipped — missing columns: {missing}")

    if check_incomplete:
        B = '\033[1m'
        R = '\033[0m'
        n_deq = (df['type'] == 'deq').sum() if 'type' in df.columns else 0
        print(f"{B}Configs after filter: {len(df)}  |  DEQ: {n_deq}  |  non-DEQ: {len(df) - n_deq}{R}")

        incomplete = []
        for _, row in df.iterrows():
            row_model_str = row[model_tag_col].replace("/", "_")
            if 'type' in df.columns and row.type == 'deq':
                row_cs = ''
                if forw_tol >= 0:
                    row_cs += f'tol{forw_tol}_'
                row_cs += f'_mi{forw_max_iter:02d}'
                if not implicit_diff:
                    row_cs += '_FNI'
                else:
                    if forw_tol != back_tol >= 0:
                        row_cs += f'_btol{back_tol}'
                    if back_max_iter != forw_max_iter:
                        row_cs += f'_bmi{back_max_iter:02d}'
                row_model_str += row_cs
            results_path = proj_path(f'{results_dir}/{row.molecule}/{row_model_str}/test.npz', use_tmp=use_tmp)
            if not os.path.exists(results_path):
                incomplete.append(row)

        incomplete_df = pd.DataFrame(incomplete)
        n_incomplete_deq = (incomplete_df['type'] == 'deq').sum() if 'type' in incomplete_df.columns and len(incomplete_df) else 0
        print(f"{B}Incomplete (no npz): {len(incomplete_df)}  |  DEQ: {n_incomplete_deq}  |  non-DEQ: {len(incomplete_df) - n_incomplete_deq}{R}")

        if tag is not None:
            out_path = proj_path(f'result_summary/runs/{tag}/{base_name}_incomplete.csv')
        else:
            out_path = proj_path('result_summary/complete_test_tmp.csv')
        incomplete_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
        return

    if array_idx is not None:
        if array_stride is not None:
            # Contiguous chunk: each worker owns a consecutive slice of the sorted array
            chunk_size = math.ceil(len(df) / array_stride)
            start = array_idx * chunk_size
            df = df.iloc[start:start + chunk_size].copy()
        else:
            df = df.iloc[[array_idx]].copy()
    else:
        df = df.copy()

    if reverse:
        df = df.iloc[::-1].reset_index(drop=True)

    # ── Column naming for test results ────────────────────────────────────────
    # Build what config_str would be for a DEQ row at the current settings,
    # then strip _mi10 / _bmi10 when they equal the default (10).
    col_prefix = model_tag_col.removesuffix('_tag')  # e.g. 'best_manual'
    _cs = ''
    if forw_tol >= 0:
        _cs += f'tol{forw_tol}_'
    _cs += f'_mi{forw_max_iter:02d}'
    if not implicit_diff:
        _cs += '_FNI'
    else:
        if forw_tol != back_tol >= 0:
            _cs += f'_btol{back_tol}'
        if back_max_iter != forw_max_iter:
            _cs += f'_bmi{back_max_iter:02d}'
    # display_meta_prefix: strips _mi10/_bmi10 at default — used only for the save filename.
    # deq column names keep the full suffix so they never collapse onto base_* columns.
    full_deq_prefix = f'test{_cs}'   # e.g. 'test_mi10', 'test_mi20_FNI'
    display_meta_prefix = full_deq_prefix
    if forw_max_iter == 10:
        display_meta_prefix = display_meta_prefix.replace('_mi10', '')
    if back_max_iter == 10:
        display_meta_prefix = display_meta_prefix.replace('_bmi10', '')
    # Non-DEQ → base_* always.  DEQ → deq_* always (always distinct from base_*).
    base_e_col = f'{col_prefix}__test_energy_MAE'
    base_f_col = f'{col_prefix}__test_forces_MAE'
    deq_e_col  = f'{col_prefix}__{full_deq_prefix}_energy_MAE'
    deq_f_col  = f'{col_prefix}__{full_deq_prefix}_forces_MAE'
    # ─────────────────────────────────────────────────────────────────────────

    api = wandb.Api()

    for i, row in tqdm(df.iloc[::1].iterrows()):
        print('='*50)
        print(i, 'of', len(df), row.type)
        print('='*50)
        model_tag = row[model_tag_col]
        model_str = model_tag.replace("/", "_")

        config_str = ''
        if row.type == 'deq':
            if forw_tol >= 0:
                config_str += f'tol{forw_tol}_'
            config_str += f'_mi{forw_max_iter:02d}'

            if not implicit_diff:
                config_str += '_FNI'
            else:
                if forw_tol != back_tol >= 0:
                    config_str += f'_btol{back_tol}'
                if back_max_iter != forw_max_iter:
                    config_str += f'_bmi{back_max_iter:02d}'

            model_str += config_str

        # else:
        #     continue

        meta_prefix = f'test{config_str}'  # e.g. 'test' or 'test__mi10'

        results_path = proj_path(f'{results_dir}/{row.molecule}/{model_str}/test.npz', use_tmp=use_tmp)
        if os.path.exists(results_path):
            print('Already exists, skipping:', results_path)
            continue

        # Try artifact metadata cache before running inference
        artifact = api.artifact(f"qcml-group/dev-v10/{model_tag}")
        if (artifact.metadata.get(f'{meta_prefix}_energy_MAE') is not None and
                artifact.metadata.get(f'{meta_prefix}_forces_MAE') is not None):
            print('Loading from artifact metadata:', results_path)
            cached = {
                k[len(meta_prefix) + 1:]: v
                for k, v in artifact.metadata.items()
                if k.startswith(meta_prefix + '_')
            }
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            np.savez(results_path, **cached)
            continue

        if csv_only:
            print('--csv-only: no cached result, leaving as missing:', results_path)
            continue

        print('Running:', results_path)

        Z = molecule_to_atomic_numbers[row.molecule]
        bsz = batch_size if batch_size is not None else int(BSZ / len(Z))
        if batch_size is None and 'bucky' in row.molecule:
            bsz = 1  # weirdly sometimes only way to avoid OOM is to start with bsz=1

        data_config = dict(
            num_train=row.num_train,
            num_val=row.num_val,
            molecule=row.molecule,
            batch_size=bsz,
            split_id=row.split_id,
            adj_cutoff=(row.cutoff or np.inf) if row.Dataset == 'MD22' else np.inf,
            duplicate_for_run=duplicate_for_run,
        )

        task = load_model_from_wandb(model_tag, return_ckpt=True)
        if row.type == 'deq':
            task.model.representation.deq.implicit_diff = implicit_diff
            task.model.representation.deq.forw_tol = forw_tol
            task.model.representation.deq.back_tol = back_tol
            task.model.representation.deq.forw_max_iter = forw_max_iter
            task.model.representation.deq.back_max_iter = back_max_iter

        def _test(dc):
            dm = _get_dm(dc)
            _callbacks = [_DEQGCCallback()] if row.type == 'deq' else []
            trainer = pl.Trainer(inference_mode=False, accelerator="auto", devices="auto",
                                 callbacks=_callbacks)
            try:
                return trainer.test(task, datamodule=dm)[0]
            finally:
                cleanup_gpu(trainer)

        try:
            test_metrics = run_with_oom_retry(_test, data_config)
        except RuntimeError:
            print('FATAL: Can not run', results_path)
            continue

        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        np.savez(results_path, **test_metrics)

        artifact.metadata.update({f'{meta_prefix}_{k}': v for k, v in test_metrics.items()})
        artifact.save()

        cleanup_gpu(task)

    # ── Read npz files → populate test metric columns → save enriched df ─────
    for col in {base_e_col, base_f_col, deq_e_col, deq_f_col}:
        df[col] = None
    for i, row in df.iterrows():
        row_model_tag = row[model_tag_col]
        row_model_str = row_model_tag.replace("/", "_")
        is_deq = row.type == 'deq'
        if is_deq:
            row_cs = ''
            if forw_tol >= 0:
                row_cs += f'tol{forw_tol}_'
            row_cs += f'_mi{forw_max_iter:02d}'
            if not implicit_diff:
                row_cs += '_FNI'
            else:
                if forw_tol != back_tol >= 0:
                    row_cs += f'_btol{back_tol}'
                if back_max_iter != forw_max_iter:
                    row_cs += f'_bmi{back_max_iter:02d}'
            row_model_str += row_cs
        results_path = proj_path(f'{results_dir}/{row.molecule}/{row_model_str}/test.npz', use_tmp=use_tmp)
        if os.path.exists(results_path):
            npz = np.load(results_path)
            e, f = float(npz['test_energy_MAE']), float(npz['test_forces_MAE'])
            if not is_deq:
                df.at[i, base_e_col] = e   # non-DEQ always in base
                df.at[i, base_f_col] = f
            df.at[i, deq_e_col] = e        # non-DEQ mirrors here too; DEQ only here
            df.at[i, deq_f_col] = f

    print(f"\n{base_f_col}: {df[base_f_col].notna().sum()}/{len(df)} populated")
    if deq_f_col != base_f_col:
        print(f"{deq_f_col}: {df[deq_f_col].notna().sum()}/{len(df)} populated")

    if tag is not None:
        has_deq = 'type' in df.columns and (df['type'] == 'deq').any()
        save_suffix = display_meta_prefix if has_deq else 'test'
        out_base = 'runs_df_val' if on_val else 'runs_best_per_group'
        out_path = proj_path(f'result_summary/runs/{tag}/{out_base}_{save_suffix}.csv')
        if os.path.exists(out_path):
            existing = pd.read_csv(out_path).set_index(model_tag_col)
            df_indexed = df.set_index(model_tag_col)
            # Union columns (existing first so original order is preserved)
            all_cols = list(existing.columns) + [c for c in df_indexed.columns if c not in existing.columns]
            existing = existing.reindex(columns=all_cols)
            # Upsert any rows from df not already in existing
            missing_rows = df_indexed.index.difference(existing.index)
            if len(missing_rows) > 0:
                new_part = df_indexed.loc[missing_rows].reindex(columns=all_cols)
                existing = pd.concat([existing, new_part])
            # Overwrite with fresh values from df wherever df has a non-NaN
            existing.update(df_indexed)
            existing.reset_index().to_csv(out_path, index=False)
        else:
            df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
    # ─────────────────────────────────────────────────────────────────────────


def _default_tag() -> str | None:
    from datetime import datetime, timedelta
    runs_base = proj_path('result_summary/runs')
    if not os.path.isdir(runs_base):
        return None
    for days_back in range(14):
        day = datetime.today() - timedelta(days=days_back)
        prefix = day.strftime("%Y-%m-%d") + '_'
        dirs = sorted(
            d for d in os.listdir(runs_base)
            if os.path.isdir(os.path.join(runs_base, d)) and d.startswith(prefix)
        )
        if dirs:
            return dirs[-1]
    return None


if __name__ == '__main__':
    # Positional order matches test-multiple.sh: TASK_ID NUM_TASKS_TOTAL [EXTRA_ARGS]
    default = _default_tag()
    parser = argparse.ArgumentParser()
    parser.add_argument('array_idx', type=int, nargs='?', default=None,
                        help='Worker index (TASK_ID from test-multiple.sh)')
    parser.add_argument('array_stride', type=int, nargs='?', default=None,
                        help='Total number of workers (NUM_TASKS_TOTAL from test-multiple.sh)')
    parser.add_argument('--tag', type=str, default=default,
                        help=f'Runs directory tag (e.g. 2026-03-07_a, default: {default}). '
                             'If omitted, uses result_summary/complete_test_tmp.csv. '
                             'Pass via EXTRA_ARGS in test-multiple.sh.')
    parser.add_argument('--max-worsening', type=float, default=2.,
                        help='Skip rows where val_forces_MAE > row_logged_min * max_worsening (default: 1.05)')
    parser.add_argument('--check', action='store_true',
                        help='Only check which filtered rows lack npz files; print summary and save incomplete df')
    parser.add_argument('--reverse', action='store_true',
                        help='Reverse row order within the already-picked chunk')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Force a specific batch size (overrides BSZ-based auto-sizing)')
    parser.add_argument('--on_val', action='store_true',
                        help='Read from runs_df_val.csv and merge results back into it '
                             '(instead of runs_best_per_group.csv)')
    parser.add_argument('--csv-only', action='store_true',
                        help='Do not run inference for rows missing a cached result; just '
                             're-merge the CSV from whatever npz/artifact-metadata cache already '
                             'exists, leaving still-missing rows as NaN in the new columns.')
    args = parser.parse_args()
    run_test(tag=args.tag, array_idx=args.array_idx, array_stride=args.array_stride,
             max_worsening=args.max_worsening, check_incomplete=args.check, reverse=args.reverse,
             batch_size=args.batch_size, on_val=args.on_val, csv_only=args.csv_only)

