import argparse
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import wandb
from tqdm import tqdm

from imlff.common.util import proj_path

_METRICS = ['val_forces_MAE', 'val_energy_MAE', 'val_loss']
_COL_PREFIX = 'val_min_logged__'


def _file_hash(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def _get_min_metrics(api, run_id: str) -> dict:
    run = api.run(f'qcml-group/dev-v10/{run_id}')
    mins = {k: float('inf') for k in _METRICS}
    found = {k: False for k in _METRICS}
    for row in run.scan_history(keys=_METRICS):
        for k in _METRICS:
            if k in row and row[k] is not None:
                mins[k] = min(mins[k], row[k])
                found[k] = True
    return {k: (mins[k] if found[k] else None) for k in _METRICS}


def add_min_val_logged(tag: str | None = None, max_workers: int = 20) -> pd.DataFrame:
    """Fetch minimum logged val metrics from W&B history and add to runs_df(_val).csv.

    Uses runs_df_val.csv if it exists, otherwise runs_df.csv.
    Saves back to the original file only if it hasn't changed in the meantime.
    Always saves to ORGFILENAME--with_min_val_logged.csv.
    """
    if tag is None:
        tag = _default_tag()
    if tag is None:
        raise ValueError('No tag given and no matching directory found for today.')

    runs_dir = proj_path(f'result_summary/runs/{tag}')
    val_path = os.path.join(runs_dir, 'runs_df_val.csv')
    base_path = os.path.join(runs_dir, 'runs_df.csv')
    csv_path = val_path if os.path.exists(val_path) else base_path
    print(f"Reading {csv_path}")

    initial_hash = _file_hash(csv_path)
    df = pd.read_csv(csv_path)

    for k in _METRICS:
        col = f'{_COL_PREFIX}{k}'
        if col not in df.columns:
            df[col] = None

    sentinel_col = f'{_COL_PREFIX}val_forces_MAE'
    needs_fetch = df.loc[df[sentinel_col].isna(), 'run_id'].dropna().tolist()
    print(f"Fetching {len(needs_fetch)}/{len(df)} runs (rest already cached)")

    api = wandb.Api()
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_get_min_metrics, api, run_id): run_id for run_id in needs_fetch}
        for future in tqdm(as_completed(futures), total=len(futures)):
            run_id = futures[future]
            try:
                results[run_id] = future.result()
            except Exception as e:
                print(f"  Warning: failed for {run_id}: {e}")
                results[run_id] = {k: None for k in _METRICS}

    for run_id, mins in results.items():
        mask = df['run_id'] == run_id
        for k, v in mins.items():
            df.loc[mask, f'{_COL_PREFIX}{k}'] = v

    backup_path = csv_path.replace('.csv', '--with_min_val_logged.csv')
    df.to_csv(backup_path, index=False)
    print(f"Saved: {backup_path}")

    current_hash = _file_hash(csv_path)
    if current_hash == initial_hash:
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
    else:
        print(f"WARNING: {os.path.basename(csv_path)} changed since read — skipping overwrite. Results in {os.path.basename(backup_path)}")

    # Always write to runs_df.csv so a subsequent run_val picks up the columns.
    if csv_path != base_path:
        base_hash_before = _file_hash(base_path)
        base_df = pd.read_csv(base_path)
        for k in _METRICS:
            col = f'{_COL_PREFIX}{k}'
            if col in df.columns:
                base_df[col] = base_df['run_id'].map(df.set_index('run_id')[col])
        base_hash_after = _file_hash(base_path)
        if base_hash_before == base_hash_after:
            base_df.to_csv(base_path, index=False)
            print(f"Saved: {base_path}")
        else:
            print(f"WARNING: {os.path.basename(base_path)} changed since read — skipping overwrite.")

    return df


def _default_tag() -> str | None:
    today_tag = datetime.today().strftime("%Y-%m-%d")
    runs_base = proj_path('result_summary/runs')
    if not os.path.isdir(runs_base):
        return None
    dirs = sorted(
        d for d in os.listdir(runs_base)
        if os.path.isdir(os.path.join(runs_base, d)) and d.startswith(today_tag + '_')
    )
    return dirs[-1] if dirs else None


if __name__ == '__main__':
    default = _default_tag()
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default=default,
                        help=f'Directory tag, e.g. 2026-03-07_a (default: {default})')
    parser.add_argument('--workers', type=int, default=20,
                        help='Parallel W&B fetch threads (default: 20)')
    args = parser.parse_args()
    if args.tag is None:
        parser.error('No tag given and no matching directory found for today.')
    add_min_val_logged(tag=args.tag, max_workers=args.workers)
