"""MD driver: single-config one-off / debug runs and tolerance sweeps.

Assembles the per-combination task list and hands it to the shared
orchestrator ``imlff.md.run.run_md``. Config is edited inline below.

Switches:
  - ``equilibrated``:    True -> start from a thermally equilibrated system
                         (skip relax/equil); False -> relax + equilibrate.
  - ``tol_grid_search``: True -> run each tolerance as its own task (tol sweep);
                         False -> single fixed-tol run.
  - model-selection block: best-models CSV / explicit-iter list / manual.

``run_md`` (the orchestrator) is re-exported here for backward compatibility:
    from imlff.scripts.run_md import run_md   # still valid

Run with:

    python -u src/imlff/scripts/run_md.py [array_idx]
"""

import argparse
import random
from itertools import product

import pandas as pd

from imlff.common.util import proj_path
from imlff.md.run import run_md  # noqa: F401  (re-exported)
from imlff.md.stateful_extrapolator import StatefulAdamsBashforth


def main():
    print('Starting run_md.py...')

    ## Main Config ##

    dev = False  # set to string to log to other wandb project

    no_md = False
    shuffle = True
    equil_wandb = False

    seeds = [0]
    # seeds = [0, 1, 2, 3, 4]
    langevin_time_constants = [1_000, 10_000, None]
    langevin_time_constant_equilibration = None

    # System-prep mode: True -> start from a thermally equilibrated system
    # (skip relax/equil); False -> relax + equilibrate pipeline.
    equilibrated = True
    if equilibrated:
        equilibrate_strategy, relax, equil_steps = 1, False, 0
    else:
        equilibrate_strategy, relax, equil_steps = None, True, 10_000

    initial_temperature = 500
    target_temperature = 300

    tags = []

    overwrite = 'append'
    overwrite = 'yes' if dev else True  # 'yes' requires no further confirmation
    steps = 2000000

    turn_into_deq = False
    max_iter = 10
    implicit_diff = True
    calc_setup = dict(
        warmstart_configs=dict(z=1, g_bar=1),
        StatefulClass=StatefulAdamsBashforth,
    )
    warmstart_mode = 'central-difference'

    ## Model selection ##
    if True:
        # read from best-models CSV
        path = proj_path('result_summary/runs/2026-06-21_ALL_128/runs_best_per_group_test.csv')
        full_df = pd.read_csv(path)
        df = full_df.query('molecule == "ethanol" and (type == "explicit" or type == "deq-merged") and AtomisticRepresentationClass == "SO3net"')
        steps = 100000
        target_temperature = 500
        dev = 'ethanol'
        seeds = range(15)
        tags = ['june25']

        df = df.query('not best_manual_tag.isna()').reset_index(drop=True)
        models = df[['molecule', 'type', 'best_manual_tag']].values
    elif False:
        # explicit / iter models (Ac-Ala3-NHMe stability analysis)
        langevin_time_constants = [0, 100, 1_000, 10_000]
        steps = 100_000
        dev = 'ala3'
        seeds = range(15)
        tags += ['explicit']
        models = [
            ('Ac-Ala3-NHMe', 'iter', 'qcml-group/dev-v10/model-bu7zjx44:v2422'),   # i1
            ('Ac-Ala3-NHMe', 'iter', 'qcml-group/dev-v10/model-y3j2gbxj:v346'),    # i2
            ('Ac-Ala3-NHMe', 'iter', 'qcml-group/dev-v10/model-gcmp3fx6:v325'),    # i3
            ('Ac-Ala3-NHMe', 'iter', 'qcml-group/dev-v10/model-j2nx77a8:v644'),    # i4
            ('Ac-Ala3-NHMe', 'iter', 'qcml-group/dev-v10/model-55ps8a42:v825'),    # i5
        ]
    else:
        # manual model selection
        models = [('aspirin', 'deq', 'qcml-group/dev-v10/model-h3bnzuij:v479')]
        models = [('ethanol', 'deq', 'qcml-group/dev-v10/model-u34do1cf:v585')]

    ## Tolerance handling ##
    # tol_grid_search=True runs each tolerance as its own task (tol sweep);
    # False does a single fixed-tol run.
    tol_grid_search = False
    if tol_grid_search:
        seeds = range(15)
        grid = False  # True: full grid product(a, a); False: line search (t, t)
        a = (0.032, 0.01, 0.0032, 0.001, 0.00032, 0)
        a = (0.01, 0.0032, 0.001, 0.00032)
        tols = list(product(a, a)) if grid else [(t, t) for t in a]
        tags += ['tol search' if grid else 'tol search line']
    else:
        tols = [(0.01, 0.01)]

    parser = argparse.ArgumentParser()
    parser.add_argument('array_idx', type=int, nargs='?', default=None)
    parser.add_argument('array_stride', type=int, nargs='?', default=None)
    parser.add_argument('--reverse', action='store_true', default=False)
    parser.add_argument('--no-relax', action='store_true', default=False,
                        help='Skip running relax; skip combination if relax result is missing')
    parser.add_argument('--no-equil', action='store_true', default=False,
                        help='Skip running equil; if equil result is missing, also skip MD for that combination')
    parser.add_argument('--no-md', action='store_true', default=False,
                        help='Skip MD; still run relax and equil')
    args = parser.parse_args()

    array_idx = args.array_idx
    reverse = args.reverse
    no_relax = args.no_relax
    no_equil = args.no_equil
    no_md = no_md or args.no_md

    if tol_grid_search:
        models_and_tolerances_to_try = list(product(models, [[t] for t in tols]))
    else:
        models_and_tolerances_to_try = [(r, tols) for r in models]

    all_combinations = list(product(models_and_tolerances_to_try, langevin_time_constants, seeds))

    print('all_combinations:')
    print(all_combinations)

    if shuffle:
        rng = random.Random(43)
        rng.shuffle(all_combinations)

    run_md(
        all_combinations=all_combinations,
        array_idx=array_idx,
        reverse=reverse,
        no_relax=no_relax,
        no_equil=no_equil,
        no_md=no_md,
        relax=relax,
        equil_steps=equil_steps,
        dev=dev,
        overwrite=overwrite,
        tol_grid_search=tol_grid_search,
        steps=steps,
        implicit_diff=implicit_diff,
        calc_setup=calc_setup,
        warmstart_mode=warmstart_mode,
        turn_into_deq=turn_into_deq,
        max_iter=max_iter,
        tags=tags,
        langevin_time_constant_equilibration=langevin_time_constant_equilibration,
        models=models,
        equil_wandb=equil_wandb,
        equilibrate_strategy=equilibrate_strategy,
        target_temperature=target_temperature,
        initial_temperature=initial_temperature,
    )


if __name__ == '__main__':
    main()
