import os
import json
import argparse
from datetime import datetime

import pandas as pd

from imlff.analysis.base import fetch_runs_df, fix_wandb_df
from imlff.common.util import proj_path


# ── Parameters ───────────────────────────────────────────────────────────────

AtomisticRepresentationClass = deq = n_z = None
# deq = True
# AtomisticRepresentationClass = 'PaiNN'
# AtomisticRepresentationClass = 'SO3net'
# AtomisticRepresentationClass = 'SchNet'
# AtomisticRepresentationClass = None
AtomisticRepresentationClass = {'$in': ['PaiNN', 'SchNet', 'SO3net']}
# AtomisticRepresentationClass = 'PaiNN'

# n_z = 128 * 2 * 2
n_z = 128

params = dict(
    # max_forw_res_10 = 0.01, # we filter this now at artifact validation
    max_forw_res_10 = None,
    allow_running = True,
    
    project_name='dev-v10',
    AtomisticRepresentationClass=AtomisticRepresentationClass,

    n_z=n_z,
    use_deq=deq,
    norm_o=deq,

    # molecule="ethanol",
    # molecule="stachyose",
    # created_within=70 * 24,
    # created_within=2000,
    # created_within=4000,
    # state={'$ne': 'running'},
    # shared_filters=True,

    Dataset={'$in': ['MD17', 'MD22']},
    
    # tags=['continue fpc feb 17'],
    
    # tags={
    #     '$nin': [
    #         # 'param search',
    #         'UNFINISHED',
    #     ],
    # },
    # molecule="aspirin",
    # InvNormClass="InvUnitLengthFeaturesNorm",
    # EquiNormClass="EquiUnitLengthFeaturesNorm",
    # MergedNormClass={'$eq': None},
    # MergedNormClass={'$ne': None},
    
    # max_runs=1,
) | {
    "summary_metrics.epoch": {'$gt': 4}
}

# ─────────────────────────────────────────────────────────────────────────────


def _postprocess(runs_df_full: pd.DataFrame, max_forw_res_10 = None, allow_running=False) -> pd.DataFrame:
    if not allow_running:
        print(f'Filtering out running:')
        print(f'{len(runs_df_full)} -> ', end='')
        runs_df_full = runs_df_full[runs_df_full.state != 'running']
        print(f'{len(runs_df_full)}')
        
    runs_df = fix_wandb_df(runs_df_full, verbose=0)
    
    print('Filter types:')
    print(f'From {len(runs_df)}:')
    print(runs_df.type.value_counts())
    runs_df = runs_df[
        runs_df.type.isin(['deq', 'deq-merged', 'explicit', 'explicit-tied', 'explicit-tt'])
    ]
    print(f'--> to {len(runs_df)}')
    print(runs_df.type.value_counts())
    
    if max_forw_res_10 is not None and 'val_forw_res_10' in runs_df:
        print(f'Filtering with {max_forw_res_10=}:')
        print(f'{len(runs_df)} -> ', end='')
        runs_df = runs_df[
              (runs_df.val_forw_res_10 <= max_forw_res_10)
            ^ (runs_df.type != 'deq')
        ]
        print(f'{len(runs_df)}')

    return runs_df


def _parse_tag_date(tag: str) -> datetime:
    """Parse '2026-03-07_d' → datetime(2026, 3, 7)."""
    date_part = tag.split('_')[0]
    return datetime.strptime(date_part, "%Y-%m-%d")


def fetch_and_save_runs(max_forw_res_10=0.01, allow_running=False, prev_tag: str | None = None, **wandb_params: dict) -> str:
    """Fetch runs from W&B, postprocess, save to a new dated directory.

    Returns the tag string (e.g. '2026-03-07_a') for downstream scripts.
    """
    if prev_tag is not None:
        base_dir = proj_path('result_summary/runs')
        prev_dir = os.path.join(base_dir, prev_tag)

        with open(os.path.join(prev_dir, 'wandb_parameters.json')) as f:
            prev_params = json.load(f)
        max_forw_res_10 = prev_params.pop('max_forw_res_10', None)
        allow_running = prev_params.pop('allow_running', None)
        wandb_params = prev_params

        cutoff = _parse_tag_date(prev_tag)
        filtered_params = wandb_params | {'heartbeat_at': {'$gte': cutoff.isoformat()}}
        runs_df = _postprocess(fetch_runs_df(**filtered_params, fix=False), max_forw_res_10=max_forw_res_10, allow_running=allow_running)

        prev_csv_df = pd.read_csv(os.path.join(prev_dir, 'runs_df.csv'))
        runs_df = pd.concat([runs_df, prev_csv_df], ignore_index=True)
        runs_df = runs_df.drop_duplicates(subset=['run_id'], keep='first')
        print(f"After merging with prev tag '{prev_tag}': {len(runs_df)} runs")
    else:
        runs_df = _postprocess(fetch_runs_df(**wandb_params, fix=False), max_forw_res_10=max_forw_res_10, allow_running=allow_running)

    print(f"\nRuns after postprocessing: {len(runs_df)}")
    for col in ['AtomisticRepresentationClass', 'type', 'n_z', 'n_interactions', 'Dataset', 'molecule',
                'EquiNormClass', 'InvNormClass', 'MergedNormClass', 'norm_o', 'norm_s', 'norm_c']:
        if col in runs_df.columns:
            print(col, runs_df[col].apply(lambda x: tuple(x) if isinstance(x, list) else x).unique())

    # Output dir: result_summary/runs/[today_tag]_a, _b, _c, …
    # If prev_tag has a complex suffix (e.g. 'all-runs-no-merge'), reuse it:
    # [today_tag]_<suffix>, then _<suffix>-b, _<suffix>-c, … on collision.
    today_tag = datetime.today().strftime("%Y-%m-%d")
    base_dir = proj_path('result_summary/runs')
    prev_suffix = prev_tag.split('_', 1)[1] if prev_tag and '_' in prev_tag else None
    is_complex = prev_suffix is not None and not (len(prev_suffix) == 1 and prev_suffix.isalpha())
    if is_complex:
        candidates = [f'{today_tag}_{prev_suffix}'] + [
            f'{today_tag}_{prev_suffix}-{l}' for l in 'bcdefghijklmnopqrstuvwxyz']
    else:
        candidates = [f'{today_tag}_{l}' for l in 'abcdefghijklmnopqrstuvwxyz']
    for cand in candidates:
        out_dir = os.path.join(base_dir, cand)
        if not os.path.exists(out_dir):
            break
    tag = os.path.basename(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nSaving to: {out_dir}")

    params_path = os.path.join(out_dir, 'wandb_parameters.json')
    with open(params_path, 'w') as f:
        json.dump(wandb_params | dict(max_forw_res_10=max_forw_res_10), f, indent=2, default=str)
    print(f"  wandb_parameters.json")

    csv_path = os.path.join(out_dir, 'runs_df.csv')
    runs_df.to_csv(csv_path, index=False)
    print(f"  runs_df.csv  ({len(runs_df)} rows)")

    return tag


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('prev_tag', type=str, nargs='?', default=None,
                        help="Previous tag (e.g. 2026-03-07_d) to backfill: re-runs that tag's "
                             "query with created_at >= 7 days before the tag date, then merges.")
    args = parser.parse_args()

    tag = fetch_and_save_runs(**params, prev_tag=args.prev_tag)
    print(f"\nTag: {tag}")
