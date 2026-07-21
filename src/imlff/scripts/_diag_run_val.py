
import argparse
import glob
import math
import os
import random
import string
from datetime import datetime
from functools import lru_cache

import pandas as pd
import pytorch_lightning as pl
import torch
import wandb


def _mem():
    if not torch.cuda.is_available():
        return "no-cuda"
    a = torch.cuda.memory_allocated() / 1024**3
    r = torch.cuda.memory_reserved() / 1024**3
    return f"alloc={a:.2f}GB reserved={r:.2f}GB"


import gc

class MemLogCallback(pl.Callback):
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        gc.collect()
        torch.cuda.empty_cache()
        n_edges = batch.get('_idx_i', batch.get('idx_i', None))
        n_edges_str = f" n_edges={len(n_edges)}" if n_edges is not None else ""
        if batch_idx < 10 or batch_idx % 50 == 0:
            print(f"    [batch {batch_idx}]{n_edges_str} {_mem()}", flush=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx < 10 or batch_idx % 50 == 0:
            print(f"    [batch {batch_idx} end] {_mem()}", flush=True)

from imlff.common.util import proj_path
from imlff.data.util import load_md17_data
from imlff.loading import load_model_from_wandb
from imlff.loading.epoch_meta import get_artifact_epoch
from imlff.scripts.util import cleanup_gpu, run_with_oom_retry

_DATA_LOAD_KEYS = [
    "num_train",
    "num_val",
    "molecule",
    "split_id",
    "adj_cutoff",
]

DATA_KEYS = _DATA_LOAD_KEYS + ["batch_size"]


@lru_cache(maxsize=1)
def _load_data_cached(**kwargs):
    """Cache data module keyed on everything except batch_size."""
    return load_md17_data(**kwargs)


def _get_dm(data_config: dict):
    """Return cached data module with batch_size applied."""
    batch_size = data_config.get('batch_size', 100)
    load_kwargs = {k: v for k, v in data_config.items() if k in _DATA_LOAD_KEYS}
    load_kwargs.setdefault('num_workers', 4)
    dm = _load_data_cached(**load_kwargs)
    if dm.val_batch_size != batch_size:
        dm.val_batch_size = batch_size
        dm._val_dataloader = None  # force re-creation with updated batch_size
    return dm


_RESULT_COLS = [
    'latest_manual_tag', 'latest_manual__val_loss', 'latest_manual__val_energy_MAE', 'latest_manual__val_forces_MAE',
    'best_manual_tag',   'best_manual__val_loss',   'best_manual__val_energy_MAE',   'best_manual__val_forces_MAE',
    'latest_manual__val_forw_res_10', 'best_manual__val_forw_res_10',
]
# columns required in cache entries for DEQ runs (must be present and non-null)
_DEQ_REQUIRED_COLS = ['latest_manual__val_forw_res_10', 'best_manual__val_forw_res_10']


def _load_val_cache(runs_dir: str) -> dict:
    """Build run_id → {'_step': ..., 'cols': {result_col: val}} from:
      - runs_df_val_partial_*.csv files in runs_dir (current tag)
      - runs_df_val.csv in all sibling result_summary/runs/ directories

    Only includes rows with a non-null best_manual_tag. The cached _step lets
    consumers detect runs that have been extended since the cache was written.

    Priority (highest wins): own partials > own full CSV > sibling CSVs.
    Siblings are loaded first so they can be overridden by current-tag data.
    This matters for DEQ runs: old sibling CSVs may lack newer result columns
    (e.g. latest_manual__val_forw_res_10) and would otherwise mask good partial data.
    """
    cache = {}

    def _ingest(df):
        for _, row in df.iterrows():
            run_id = row.get('run_id')
            if pd.isna(run_id) or pd.isna(row.get('best_manual_tag')):
                continue
            cache[run_id] = {
                '_step': row.get('_step') if '_step' in df.columns else None,
                'cols': {col: row.get(col) for col in _RESULT_COLS if col in df.columns},
            }

    # Siblings first (lowest priority — overridden by own data below)
    runs_base = os.path.dirname(runs_dir)
    for other_name in sorted(os.listdir(runs_base)):
        other_dir = os.path.join(runs_base, other_name)
        if other_dir == runs_dir or not os.path.isdir(other_dir):
            continue
        full_csv = os.path.join(other_dir, 'runs_df_val.csv')
        if os.path.exists(full_csv):
            try:
                _ingest(pd.read_csv(full_csv))
                print(f"  loaded sibling: {other_name}/runs_df_val.csv")
            except Exception as e:
                print(f"  Warning: could not load {full_csv}: {e}")

    # Own full CSV (overrides siblings)
    own_csv = os.path.join(runs_dir, 'runs_df_val.csv')
    if os.path.exists(own_csv):
        try:
            _ingest(pd.read_csv(own_csv))
            print(f"  loaded own: runs_df_val.csv ({len(cache)} entries)")
        except Exception as e:
            print(f"  Warning: could not load {own_csv}: {e}")

    # Own partials last (highest priority — override everything)
    for path in sorted(glob.glob(os.path.join(runs_dir, 'runs_df_val_partial_*.csv'))):
        try:
            _ingest(pd.read_csv(path))
            print(f"  loaded partial: {os.path.basename(path)}")
        except Exception as e:
            print(f"  Warning: could not load {path}: {e}")

    return cache


def _apply_cache_entry(df: pd.DataFrame, idx, run_id, val_cache: dict) -> bool:
    """Copy cached result cols into df at idx if cache hit and not stale.

    Stale = cached _step != current df _step (run was extended since cache write).
    Returns True if cache was applied, False otherwise.
    """
    entry = val_cache.get(run_id)
    if entry is None:
        return False
    cached_step = entry.get('_step')
    current_step = df.at[idx, '_step'] if '_step' in df.columns else None
    cached_missing = cached_step is None or pd.isna(cached_step)
    current_missing = current_step is None or pd.isna(current_step)
    if cached_missing or current_missing or cached_step != current_step:
        print(f"  (cache stale: cached _step={cached_step} vs current _step={current_step}, re-evaluating)")
        return False
    use_deq = bool(df.at[idx, 'use_deq']) if 'use_deq' in df.columns else False
    if use_deq:
        for c in _DEQ_REQUIRED_COLS:
            v = entry['cols'].get(c)
            if v is None or pd.isna(v):
                print(f"  (cache missing {c} for deq run, re-evaluating)")
                return False
    for col, val in entry['cols'].items():
        df.at[idx, col] = val
    return True


def validate_artifacts(
    tag: str,
    array_idx: int | None = None,
    array_stride: int | None = None,
    reverse: bool = False,
    max_worsening: float = 1.05,
    batch_size: int | None = None,
    rerun: bool = False,
    refetch: bool = False,
    max_last_res: float = 0.01,
    max_artifacts: int | None = None,
) -> pd.DataFrame:
    """For each run in runs_df, evaluate all model artifacts on the val set.

    Workers (array_idx given) save a partial CSV instead of the full one.
    A full run (array_idx=None) saves runs_df_val.csv and deletes partials.
    """
    today_tag = datetime.today().strftime("%Y-%m-%d")
    runs_dir = proj_path(f'result_summary/runs/{tag}')
    complete_csv = os.path.join(runs_dir, 'complete_val.csv')
    fallback_csv = os.path.join(runs_dir, 'runs_df.csv')
    if array_idx is not None and os.path.exists(complete_csv):
        csv_path = complete_csv
    else:
        csv_path = fallback_csv
    print(f"Reading {csv_path}")
    runs_df = pd.read_csv(csv_path)
    print(f"Loaded {len(runs_df)} runs")

    flag_notes = []
    if rerun:
        flag_notes.append("--rerun: ignoring CSV/metadata caches, re-validating every artifact")
    if refetch:
        flag_notes.append("--refetch: ignoring CSV cache, but artifact metadata cache still honored")
    if reverse:
        flag_notes.append("--reverse: reversed run + artifact order")
    if batch_size is not None:
        flag_notes.append(f"--batch-size {batch_size}: overriding per-run batch size")
    if max_worsening != 1.05:
        flag_notes.append(f"--max-worsening {max_worsening}: stricter logged-min filter for select_best_runs")
    if max_last_res != 0.01:
        flag_notes.append(f"--max-last-res {max_last_res}: artifacts with val_forw_res_10 > {max_last_res} are not eligible as best")
    if array_idx is not None:
        flag_notes.append(f"array_idx={array_idx} array_stride={array_stride}: worker mode (saves partial CSV)")
    for note in flag_notes:
        print(note)

    # Copy val_min_logged__ columns from runs_df_val.csv if available and not already present
    val_csv_path = os.path.join(runs_dir, 'runs_df_val.csv')
    if os.path.exists(val_csv_path):
        try:
            val_df = pd.read_csv(val_csv_path)
            mi_cols = [c for c in val_df.columns if c.startswith('val_min_logged__')
                       and c not in runs_df.columns]
            if mi_cols:
                runs_df = runs_df.merge(val_df[['run_id'] + mi_cols], on='run_id', how='left')
                print(f"Copied {mi_cols} from runs_df_val.csv")
        except Exception as e:
            print(f"Warning: could not copy val_min_logged__ from runs_df_val.csv: {e}")

    api = wandb.Api()

    # Sort by data config so adjacent rows share the same data loader
    sort_cols = [k for k in DATA_KEYS if k in runs_df.columns]
    runs_df = runs_df.sort_values(sort_cols).reset_index(drop=True)

    if reverse:
        runs_df = runs_df.iloc[::-1]
        direction = 'rev'
    else:
        direction = 'fwd'
        
    if array_idx is not None:
        if array_stride is not None:
            chunk_size = math.ceil(len(runs_df) / array_stride)
            start = array_idx * chunk_size
            runs_df = runs_df.iloc[start:start + chunk_size].copy()
        else:
            runs_df = runs_df.iloc[[array_idx]].copy()
            start = array_idx
        print(f"Worker {array_idx}/{array_stride} ({direction}): rows {start}–{start + len(runs_df) - 1} ({len(runs_df)} runs)")

    rand_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    partial_path = os.path.join(runs_dir, f'runs_df_val_partial_{"full_" * (array_idx is None)}{direction}_{rand_suffix}.csv')

    if rerun or refetch:
        val_cache = {}
    else:
        print("Loading cached val metrics from partial/sibling CSVs...")
        val_cache = _load_val_cache(runs_dir)
        print(f"Pre-loaded {len(val_cache)} runs from CSV cache")

    for col in _RESULT_COLS:
        runs_df[col] = None

    n = len(runs_df)
    processed_rows = []
    try:
        for i, row in enumerate(runs_df.itertuples()):
            idx = row.Index
            run_id = row.run_id
            print(f"\n({i+1}/{n}) run {run_id}")

            if not (rerun or refetch) and _apply_cache_entry(runs_df, idx, run_id, val_cache):
                print(f"  (using CSV cache)")
                processed_rows.append(idx)
                continue

            data_config = {k: v for k, v in row._asdict().items() if k in DATA_KEYS}
            if data_config['adj_cutoff'] == 'Infinity':
                data_config['adj_cutoff'] = float('inf')
            if batch_size is not None:
                data_config['batch_size'] = batch_size
            print(data_config)
            print(f"Number of conformations: {len(_get_dm(data_config).val_dataloader())}")

            run = api.run(f"qcml-group/dev-v10/{run_id}")
            artifacts = list(run.logged_artifacts())
            if reverse:
                artifacts = list(reversed(artifacts))
            print(f"Artifacts for run {run_id}:")

            latest_manual_v = -1
            latest_manual = None
            latest_manual_metrics = {}
            best_manual__val_error = float('inf')
            best_manual = None
            best_manual_metrics = {}
            oom_skip = False
            artifact_count = 0

            for artifact in artifacts:
                if 'model-' not in artifact.name:
                    continue
                aliases = artifact.aliases
                print(f"  - {artifact.name} (aliases: {aliases})")

                use_deq = bool(getattr(row, 'use_deq', False))
                meta_cache_ok = artifact.metadata.get('val_loss_manual') is not None
                if use_deq and artifact.metadata.get('val_forw_res_10_manual') is None:
                    meta_cache_ok = False
                if not rerun and meta_cache_ok:
                    print(f"    (using cached metadata)")
                    metrics = {k[:-len('_manual')]: v for k, v in artifact.metadata.items() if k.endswith('_manual')}
                else:
                    artifact_count += 1
                    if max_artifacts is not None and artifact_count > max_artifacts:
                        print(f"  [diag] reached max_artifacts={max_artifacts}, stopping")
                        break
                    artifact_fqn = "/".join([artifact.entity, artifact.project, artifact.source_name])
                    print(f"  [before load] {_mem()}", flush=True)
                    try:
                        task = load_model_from_wandb(artifact_fqn, return_ckpt=True, as_deq_task=use_deq)
                    except (ValueError, wandb.errors.CommError) as e:
                        print(f"  WARNING: skipping {artifact.name} — unavailable (deleted / running run mid-upload?): {e}")
                        continue
                    print(f"  [after load] {_mem()}", flush=True)
                    if use_deq:
                        from imlff.model.deq import DEQModel
                        for _name, _mod in task.named_modules():
                            if isinstance(_mod, DEQModel):
                                _mod.implicit_diff = True
                                print(f"  [diag] overrode implicit_diff=True on {_name}", flush=True)

                    def _validate(dc):
                        dm = _get_dm(dc)
                        trainer = pl.Trainer(inference_mode=False, accelerator="auto", devices="auto",
                                             callbacks=[MemLogCallback()])
                        try:
                            return trainer.validate(task, datamodule=dm)[0]
                        finally:
                            cleanup_gpu(trainer)

                    try:
                        val_metrics = run_with_oom_retry(_validate, data_config)
                    except RuntimeError as e:
                        if 'OOM even at batch_size=1' in str(e):
                            print(f"  WARNING: skipping run {run_id} — OOM at batch_size=1")
                            cleanup_gpu(task)
                            oom_skip = True
                            break
                        raise
                    print(f"  [before task cleanup] {_mem()}", flush=True)
                    cleanup_gpu(task)
                    print(f"  [after task cleanup] {_mem()}", flush=True)

                    # drop `<x>_max` only if `<x>` exists with the same value (else keep both, or keep `_max` alone)
                    for m in list(val_metrics.keys()):
                        if '_max' in m:
                            n = m.replace('_max', '')
                            if n in val_metrics and val_metrics[n] == val_metrics[m]:
                                val_metrics.pop(m)

                    # keep only `val_forw_res_<NN>` among `_res_` keys (drop val_back_res_*, etc.)
                    for m in list(val_metrics.keys()):
                        if '_res_' in m and m[:-2] != 'val_forw_res_':
                            val_metrics.pop(m)

                    def _clean(v):
                        try:
                            v = v.item() if hasattr(v, 'item') else v
                            return None if v is not None and math.isnan(v) else v
                        except TypeError:
                            return v

                    metrics = {k: _clean(v) for k, v in val_metrics.items()}
                    artifact.metadata.update({f'{k}_manual': v for k, v in metrics.items()})
                    artifact.save()

                val_loss = metrics.get('val_loss')
                val_energy_MAE = metrics.get('val_energy_MAE')
                val_forces_MAE = metrics.get('val_forces_MAE')

                # Pick latest_manual by AUTHORITATIVE ckpt epoch, not by the
                # `:vNN` version number (monotonic by upload order, not by
                # training epoch — a resumed run can upload a low-epoch ckpt
                # AFTER a high-epoch one). get_artifact_epoch backfills
                # metadata['epoch'] for legacy artifacts via torch.load.
                v = get_artifact_epoch(artifact, backfill=True)
                if v is None:
                    v = -1  # unrankable; never wins
                if v > latest_manual_v:
                    latest_manual_v = v
                    latest_manual = artifact
                    latest_manual_metrics = metrics

                last_res = metrics.get('val_forw_res_10')
                eligible_for_best = last_res is None or last_res <= max_last_res
                if eligible_for_best and val_forces_MAE is not None and val_forces_MAE < best_manual__val_error:
                    best_manual__val_error = val_forces_MAE
                    best_manual = artifact
                    best_manual_metrics = metrics

            if oom_skip:
                continue

            # Assign wandb alias tags
            if latest_manual is not None:
                latest_manual.aliases.append(today_tag + '-latest')
                latest_manual.save()

            if best_manual is not None:
                best_manual.aliases.append(today_tag + '-best')
                best_manual.save()

            runs_df.at[idx, 'latest_manual_tag'] = latest_manual.name if latest_manual else None
            for k, v in latest_manual_metrics.items():
                runs_df.at[idx, f'latest_manual__{k}'] = v

            runs_df.at[idx, 'best_manual_tag'] = best_manual.name if best_manual else None
            for k, v in best_manual_metrics.items():
                runs_df.at[idx, f'best_manual__{k}'] = v

            processed_rows.append(idx)
            runs_df.loc[processed_rows].to_csv(partial_path, index=False)
            print(f"  (saved partial, {len(processed_rows)} rows)")

    finally:
        if processed_rows:
            runs_df.loc[processed_rows].to_csv(partial_path, index=False)
            print(f"\nSaved partial: {partial_path}")

    if array_idx is None:
        out_path = os.path.join(runs_dir, 'runs_df_val.csv')
        tmp_path = out_path + '.tmp'
        runs_df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, out_path)  # atomic rename
        print(f"\nSaved: {out_path}")

        for path in glob.glob(os.path.join(runs_dir, 'runs_df_val_partial_*.csv')):
            os.remove(path)
            print(f"Deleted partial: {path}")

        select_best_runs(tag, runs_df=runs_df, max_worsening=max_worsening)

    return runs_df


GROUP_COLS = [
    'AtomisticRepresentationClass',
    'Dataset',
    'molecule',
    'n_z',
    'type',
    'layer_calls',
    'n_interactions',
    'shared_interactions',
    'n_schnets',
    'shared_schnets',
    'norm_o',
    'norm_s',
    'norm_c',
    'EquiNormClass',
    'InvNormClass',
    'MergedNormClass',
    'shared_filters',
]


def select_best_runs(
    tag: str,
    runs_df: pd.DataFrame | None = None,
    max_worsening: float = 1.05,
) -> pd.DataFrame:
    """Within each (molecule × architecture × type × n_interactions) group, pick
    the run with the lowest best_manual__val_forces_MAE.

    Rows where best_manual__val_forces_MAE > val_min_logged__val_forces_MAE * max_worsening
    are excluded before group selection, allowing another group member to win instead.

    If runs_df is not supplied, reads runs_df_val.csv from the tag directory.
    Saves the result as runs_best_per_group.csv and returns it.
    """
    runs_dir = proj_path(f'result_summary/runs/{tag}')

    B, R = '\033[1m', '\033[0m'

    if runs_df is None:
        val_csv = os.path.join(runs_dir, 'runs_df_val.csv')
        if os.path.exists(val_csv):
            runs_df = pd.read_csv(val_csv)
        else:
            # Fall back to partial/sibling cache so --best works mid-run
            print(f"{B}runs_df_val.csv not found — building from partial/sibling cache{R}")
            base_df = pd.read_csv(os.path.join(runs_dir, 'runs_df.csv'))
            val_cache = _load_val_cache(runs_dir)
            for col in _RESULT_COLS:
                base_df[col] = None
            for idx, row in base_df.iterrows():
                _apply_cache_entry(base_df, idx, row.run_id, val_cache)
            runs_df = base_df

    criteria_col = 'best_manual__val_forces_MAE'
    logged_col = 'val_min_logged__val_forces_MAE'
    group_cols = [c for c in GROUP_COLS if c in runs_df.columns]

    # Drop unevaluated rows; note how many groups are still partially incomplete
    if criteria_col in runs_df.columns:
        n_unevaluated = int(runs_df[criteria_col].isna().sum())
        if n_unevaluated:
            has_nan = runs_df.groupby(group_cols, dropna=False)[criteria_col].transform(lambda g: g.isna().any()).fillna(False).astype(bool)
            n_incomplete_groups = int(runs_df[has_nan].groupby(group_cols, dropna=False).ngroups)
            print(f"{B}Skipping {n_unevaluated} unevaluated runs ({n_incomplete_groups} groups still incomplete){R}")
        runs_df = runs_df[runs_df[criteria_col].notna()].copy()

    # Filter out runs that are too much worse than their own logged minimum
    eligible = runs_df.copy()
    if criteria_col in eligible.columns and logged_col in eligible.columns:
        both = eligible[criteria_col].notna() & eligible[logged_col].notna()
        too_bad = both & (eligible[criteria_col] > eligible[logged_col] * max_worsening)
        n_dropped = int(too_bad.sum())
        if n_dropped:
            print(f"{B}max_worsening={max_worsening}: excluded {n_dropped}/{len(eligible)} runs before group selection{R}")
        eligible = eligible[~too_bad]
    else:
        missing = [c for c in [criteria_col, logged_col] if c not in eligible.columns]
        print(f"{B}max_worsening filter skipped — missing columns: {missing}{R}")

    print(f"Group cols: {group_cols}")
    print(f"Criteria:   {criteria_col}")

    idx = eligible.groupby(group_cols, dropna=False)[criteria_col].idxmin()
    best_per_group = eligible.loc[idx].sort_values(group_cols).reset_index(drop=True)

    print(f"\nGroups: {len(best_per_group)}")
    for col in group_cols:
        print(f"  {col}: {sorted(best_per_group[col].dropna().unique().tolist())}")

    out_path = os.path.join(runs_dir, 'runs_best_per_group.csv')
    best_per_group.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    if 'type' in best_per_group.columns:
        deq_rows = best_per_group[best_per_group['use_deq']]
        if len(deq_rows) and 1:
            warmstart_path = proj_path('result_summary/complete_warmstart_analysis_warm_tmp.csv')
            deq_rows.to_csv(warmstart_path, index=False)
            print(f"{B}Saved {len(deq_rows)} DEQ rows → {warmstart_path}{R}")

    return best_per_group


def _consolidate_partials(runs_dir: str) -> str | None:
    """Merge all runs_df_val_partial_*.csv files in runs_dir into one new partial
    with a random suffix, then delete the originals.

    Sorted ingestion + drop_duplicates(keep='last') preserves the same
    "alphabetically later partial wins" precedence used by _load_val_cache.
    Returns the new partial path, or None if no partials existed.
    """
    partials = sorted(glob.glob(os.path.join(runs_dir, 'runs_df_val_partial_*.csv')))
    if not partials:
        return None
    dfs = []
    for p in partials:
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            print(f"  Warning: could not load {p}: {e}")
    if not dfs:
        return None
    merged = pd.concat(dfs, ignore_index=True)
    if 'run_id' in merged.columns:
        merged = merged.drop_duplicates(subset='run_id', keep='last')
    rand_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    out_path = os.path.join(runs_dir, f'runs_df_val_partial_merged_{rand_suffix}.csv')
    merged.to_csv(out_path, index=False)
    print(f"Consolidated {len(partials)} partials → {os.path.basename(out_path)} ({len(merged)} rows)")
    for p in partials:
        if p == out_path:
            continue
        os.remove(p)
        print(f"  Deleted: {os.path.basename(p)}")
    return out_path


def check_val(tag: str) -> pd.DataFrame:
    """Show completion status across all workers for a tag and save complete_val.csv.

    Reads runs_df.csv, merges all cached val metrics (partials + siblings),
    prints a bold summary, and saves complete_val.csv in the tag directory.
    If work is incomplete, consolidates all partials into a single new partial.
    """
    B, R = '\033[1m', '\033[0m'
    runs_dir = proj_path(f'result_summary/runs/{tag}')
    runs_df = pd.read_csv(os.path.join(runs_dir, 'runs_df.csv'))

    print("Loading cached val metrics from partial/sibling CSVs...")
    val_cache = _load_val_cache(runs_dir)
    print(f"Pre-loaded {len(val_cache)} runs from CSV cache")

    for col in _RESULT_COLS:
        runs_df[col] = None
    for idx, row in runs_df.iterrows():
        _apply_cache_entry(runs_df, idx, row.run_id, val_cache)

    done = runs_df['best_manual_tag'].notna()
    n_total = len(runs_df)
    n_done = done.sum()
    n_missing = n_total - n_done

    is_deq = runs_df['type'] == 'deq' if 'type' in runs_df.columns else pd.Series(False, index=runs_df.index)
    is_deq_merged = runs_df['type'] == 'deq-merged' if 'type' in runs_df.columns else pd.Series(False, index=runs_df.index)
    is_other = ~(is_deq | is_deq_merged)

    n_deq_total, n_deq_done = is_deq.sum(), (done & is_deq).sum()
    n_dm_total, n_dm_done = is_deq_merged.sum(), (done & is_deq_merged).sum()
    n_other_total, n_other_done = is_other.sum(), (done & is_other).sum()

    print(f"{B}Total:   {n_total:4d}  |  deq: {n_deq_total}  |  deq-merged: {n_dm_total}  |  other: {n_other_total}{R}")
    print(f"{B}Done:    {n_done:4d}  |  deq: {n_deq_done}  |  deq-merged: {n_dm_done}  |  other: {n_other_done}{R}")
    print(f"{B}Missing: {n_missing:4d}  |  deq: {n_deq_total - n_deq_done}  |  deq-merged: {n_dm_total - n_dm_done}  |  other: {n_other_total - n_other_done}{R}")

    incomplete = runs_df[~done].reset_index(drop=True)
    out_path = os.path.join(runs_dir, 'complete_val.csv')
    incomplete.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(incomplete)} incomplete rows)")

    if n_missing > 0:
        _consolidate_partials(runs_dir)

    return runs_df


def _default_tag() -> str | None:
    from datetime import timedelta
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
    default = _default_tag()
    parser = argparse.ArgumentParser()
    parser.add_argument('array_idx', type=int, nargs='?', default=None,
                        help='Worker index (for parallel runs via test-multiple.sh)')
    parser.add_argument('array_stride', type=int, nargs='?', default=None,
                        help='Total number of workers')
    parser.add_argument('--tag', type=str, default=default,
                        help=f'Directory tag, e.g. 2026-03-07_a (default: {default})')
    parser.add_argument('--best', action='store_true',
                        help='Recompute runs_best_per_group.csv from current results (works mid-run via cache)')
    parser.add_argument('--check', action='store_true',
                        help='Show completion status across all workers and save complete_val.csv')
    parser.add_argument('--reverse', action='store_true',
                        help='Reverse run order (after chunk slicing) and artifact iteration order')
    parser.add_argument('--max-worsening', type=float, default=1.05,
                        help='Exclude runs where val_forces_MAE > logged_min * max_worsening before group selection (default: 1.05)')
    parser.add_argument('--max-last-res', type=float, default=0.01,
                        help='Artifacts with val_forw_res_10 above this are not eligible as best (default: 0.01)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Force a specific batch size (overrides per-run config)')
    parser.add_argument('--rerun', action='store_true',
                        help='Re-validate every artifact, ignoring CSV and per-artifact metadata caches; metadata is augmented (existing keys overwritten, others preserved)')
    parser.add_argument('--refetch', action='store_true',
                        help='Like --rerun for CSV caches (ignored), but artifact metadata cache is still honored — useful for repopulating runs_df from existing artifact metadata')
    parser.add_argument('--max-artifacts', type=int, default=None,
                        help='[diag] Stop after this many actual validations per run (skips cached)')
    args = parser.parse_args()
    if args.tag is None:
        parser.error('No tag given and no matching directory found for today.')
    if args.best:
        select_best_runs(args.tag, max_worsening=args.max_worsening)
    elif args.check:
        check_val(args.tag)
    else:
        validate_artifacts(args.tag, array_idx=args.array_idx, array_stride=args.array_stride, reverse=args.reverse, max_worsening=args.max_worsening, batch_size=args.batch_size, rerun=args.rerun, refetch=args.refetch, max_last_res=args.max_last_res, max_artifacts=args.max_artifacts)