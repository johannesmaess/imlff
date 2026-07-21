"""Shared MD orchestrator extracted from ``scripts/run_md.py``.

This module owns the ``run_md(...)`` function — the per-combination loop that
each ``scripts/md_<usecase>.py`` invokes after assembling its own
``all_combinations`` list. Behavior is a literal extract from
``scripts/run_md.py`` as of the 2026-05-23 split; no logic changes.

The only externally-visible delta vs. the pre-split version: ``Init`` is now a
keyword argument (it used to be captured from the script's module-level scope).
Defaults to ``UniformInit`` to match prior behavior.
"""

import os
import sys
import pickle

import numpy as np
import wandb
from filelock import FileLock

from schnetpack.md.integrators import VelocityVerlet
from schnetpack.md.initial_conditions import UniformInit

from imlff.loading import artifact_or_run_cached
from imlff.md.logger import WandBLoggerMD
from imlff.md.util import (
    NaNCheckHook,
    BondBreakCheckHook,
    get_next_path,
    desc_md,
    desc_molecule,
    load_checkpoint,
    load_update_desc_model,
    load_relax_equil_atoms,
    load_system_equilibrated,
    setup_filesystem_hooks,
    setup_md,
)
from imlff.model.platform import wait_for_cuda
from imlff.common.util import proj_path
from imlff.model.solvers import fixed_point_iteration
from imlff.version import VERSION
from imlff.data.info import (
    md17_conformations_per_molecule,
    md22_conformations_per_molecule,
)


def run_md_tol_loop(
    *,
    prepare_system,
    desc_molecule_str,
    model,
    model_tag,
    molecule_name,
    molecule_config,
    md_config,
    seed,
    use_deq,
    tolerances_to_try,
    no_md,
    dev,
    device,
    overwrite,
    tol_grid_search,
    steps,
    implicit_diff,
    calc_setup,
    warmstart_mode,
    max_iter,
    tags,
    training_config,
    log_properties,
    bonds_and_stats_per_molecule,
):
    """Per-(forw_tol, back_tol) MD loop, shared by both system-prep branches of
    ``run_md`` (relax+equil vs. the ``load_system_equilibrated`` variant).

    The only thing that differs between callers is how the system is prepared
    (relax+equil vs. ``load_system_equilibrated``) and the ``desc_molecule``
    string used to build the results path. Both are injected:

    - ``prepare_system``: a zero-arg callable returning the prepared system
      (ASE ``Atoms`` or SchNetPack ``System``). It is called once per tol
      iteration, before MD, and is responsible for its own equil-cache lock.
    - ``desc_molecule_str``: the molecule path segment under ``results/``.

    Body extracted verbatim from ``run_md``'s inner loop (2026-06-06); no
    logic changes.
    """
    for forw_tol, back_tol in tolerances_to_try if use_deq else [(None, None)]:
        system = prepare_system()
        if no_md:
            break

        if use_deq:
            print('=' * 80)
            print('Attempting to run MD with tolerances:', forw_tol, back_tol)
            print('=' * 80)

            wc = calc_setup['warmstart_configs']
            if 'lead' in calc_setup:
                wc = wc[calc_setup['lead']]

            model_config = dict(
                model_tag=model_tag,
                implicit_diff=implicit_diff,
                forw_tol=forw_tol,
                back_tol=back_tol,
                forw_solver=fixed_point_iteration,
                back_solver=fixed_point_iteration,
                forw_max_iter=max_iter,
                back_max_iter=max_iter,
                warmstart_configs=wc,
            )
        else:
            model_config = dict(model_tag=model_tag)

        model, model_config_description = load_update_desc_model(model=model, device=device, **model_config)
        md_path = proj_path('results-'+str(dev) if dev else 'results', desc_molecule_str, model_config_description)
        md_path = get_next_path(md_path)
        md_name = desc_md(**md_config)
        log_file = f"{md_path}/{md_name}"
        path_in_proj = log_file.split('imlff/')[-1]

        log_lock = FileLock(log_file + ".lock")
        with log_lock:
            print('log_file:\n', log_file)
            print('\nload this setting in the future with:\n')
            print(f"log_file = proj_path('{path_in_proj}')\n\n")

            hooks = setup_filesystem_hooks(log_file, overwrite, log_properties=None) + [NaNCheckHook(), BondBreakCheckHook(bonds_and_stats=bonds_and_stats_per_molecule[molecule_name], k_termination=np.inf, r_termination=np.inf, m_termination=0.)]
            wandb_logger = WandBLoggerMD(
                project=str(dev) if dev else 'md-v7-test',
                name=md_name,
                tags=tags,
                hyperparams=dict(
                    warmstart_mode=warmstart_mode,
                    md_config=md_config,
                    model_config=model_config,
                    molecule_config=molecule_config,
                    seed=seed,
                    model_config_description=model_config_description,
                    md_version=7,
                    version=VERSION,
                    proj_path=path_in_proj,
                    train=training_config,
                    slurm_job_id=int(os.environ.get('SLURM_JOB_ID', -1)),
                ),
                log_properties=log_properties,
            )
            hooks.append(wandb_logger)
            md_simulator, _ = setup_md(
                model, system, md_config,
                log_properties=log_properties,
                hooks=hooks,
                **calc_setup,
            )
            load_checkpoint(md_simulator, log_file, overwrite)

            interrupted = False
            try:
                print('Running MD...')
                md_simulator.simulate(steps)
            except (KeyboardInterrupt, SystemExit):
                print('Simulation interrupted by user.')
                interrupted = True
            except ValueError as e:
                known_errors = [
                    'NaN detected in atomic positions.',
                    "Bond breaking detected."
                ]
                for ke in known_errors:
                    if ke in str(e):
                        print(ke)
                        print(e)
                        break
                else:
                    raise e
            finally:
                # log total_time and time_per_step even on exception
                wandb_logger.log_timing(md_simulator)
                if wandb_logger.run is not None:
                    wandb_logger.run.finish()
                    wandb_logger.run = None
                print(
                    f'total_time={wandb_logger.total_time} s, '
                    f'time_per_step={wandb_logger.time_per_step} s'
                )

            if interrupted:
                break

            if md_simulator.step == steps:
                print(f"Simulation finished successfully after {steps} steps.")
                if not tol_grid_search:
                    break
            else:
                print('\n' * 20)


def run_md(
    all_combinations,
    *,
    array_idx,
    reverse,
    no_relax,
    no_equil,
    no_md,
    relax,
    equil_steps,
    dev,
    overwrite,
    tol_grid_search,
    steps,
    implicit_diff,
    calc_setup,
    warmstart_mode,
    turn_into_deq,
    max_iter,
    tags,
    langevin_time_constant_equilibration,
    models,
    equil_wandb,
    Init=UniformInit,
    equilibrate_strategy=None,
    target_temperature=300,
    initial_temperature=500,
):
    """Per-combination MD loop.

    System preparation follows one of three branches, selected per call:

    - ``equilibrate_strategy`` (an int): skip geometry relaxation and bring the
      raw atoms straight to thermal equilibrium via ``load_system_equilibrated``.
      ``target_temperature`` / ``initial_temperature``
      (defaults 300 / 500) drive the equilibration and the production MD. When
      set, this branch wins and all relax/equil parameters are forced off so
      they log as falsy/None to wandb (``relax=False``, ``equil_steps=0``,
      ``model_config_relax=None``, ``equil_md_config=None``,
      ``langevin_time_constant_equilibration=None``).
    - ``relax`` (truthy, ``equilibrate_strategy`` unset): the relax ->
      equilibrate pipeline via ``load_relax_equil_atoms``.
    - neither: no system preparation.
    """
    if equilibrate_strategy is not None:
        # The equilibrate branch owns system prep; force the relax/equil
        # pipeline fully off so nothing stale is logged to wandb.
        relax = False
        equil_steps = 0
        langevin_time_constant_equilibration = None

    stats_path = proj_path('data', "bonds_and_stats.pkl")
    with open(stats_path, "rb") as f:
        bonds_and_stats_per_molecule = pickle.load(f)

    print('array_idx:', array_idx)
    print('num models:', len(models))
    print('num tasks:', len(all_combinations))

    # Now iterate with unified counter
    for i, (((molecule_name, model_type, model_tag), tolerances_to_try), langevin_time_constant, seed) in enumerate(all_combinations):

        if equilibrate_strategy is not None:
            md_config = dict(
                timestep=0.5,
                temperature=target_temperature,
                integrator=VelocityVerlet,
                langevin_time_constant=langevin_time_constant,
                initial_temperature=initial_temperature,
                Init=Init,
            )
        else:
            md_config = dict(
                timestep=0.5,
                temperature=target_temperature,
                integrator=VelocityVerlet,
                langevin_time_constant=langevin_time_constant,
                initial_temperature=initial_temperature*(not relax),
                Init=Init,
            )

        use_deq = 'deq' in model_type  # 'deq' and 'deq-merged' both use the DEQ path

        print(i, len(all_combinations), seed, model_tag, use_deq, langevin_time_constant, tolerances_to_try)

        if array_idx is not None and i != array_idx:
            continue

        print('='*20)

        np.random.seed(seed)
        molecule_idx = md17_conformations_per_molecule.get(molecule_name, md22_conformations_per_molecule.get(molecule_name))
        assert molecule_idx is not None, f"Unknown molecule: {molecule_name}"
        molecule_idx = np.random.randint(0, molecule_idx)

        if model_tag.startswith('model-'):
            model_tag = artifact_or_run_cached(model_tag, return_tag=True)

        # Per-combination overrides (reset each iteration)
        _equil_steps = equil_steps
        _no_md = no_md

        device = wait_for_cuda(0.5 if dev else 30) if sys.platform != "darwin" else 'cpu'
        if device == 'cuda':
            print('CUDA available!')
        else:
            print('CUDA not available!')

        skip_combination = False
        if equilibrate_strategy is not None:
            # Equilibrate branch: skip geometry relaxation; bring raw atoms to
            # thermal equilibrium via load_system_equilibrated. relax/equil
            # params were forced off above so they log falsy/None to wandb.

            # Equilibration uses its own fixed config inside
            # bring_to_thermal_equilibrium, so only the base model is needed.
            model, _ = load_update_desc_model(
                model=None, model_tag=model_tag, turn_into_deq=turn_into_deq, device=device,
            )

            # Logged to wandb under molecule_config.* and splatted straight into
            # bring_to_thermal_equilibrium, which takes `equilibrate_strategy`.
            molecule_config = dict(
                molecule_name=molecule_name,
                molecule_idx=molecule_idx,
                equilibrate_strategy=equilibrate_strategy,
                initial_temperature=initial_temperature,
                target_temperature=target_temperature,
                Init=Init,
            )

            equil_dir = proj_path(
                'results-dev-equil' if dev else 'results',
                'equilibrated',
                f"{molecule_name}_{molecule_idx}",
                f"s{equilibrate_strategy}_T{target_temperature:04d}K_iT{initial_temperature:04d}K",
            )
            equil_path = os.path.join(equil_dir, f"equilibrated_{model_tag.split('model-')[1]}.pt")

            desc_molecule_str = desc_molecule(
                molecule_name=molecule_name, molecule_idx=molecule_idx, fmax=0.01, stepsmax=1000,
            )

            def prepare_system(model=model, equil_path=equil_path, molecule_config=molecule_config):
                equil_lock = FileLock(equil_path + ".lock")
                with equil_lock:
                    return load_system_equilibrated(model=model, equil_path=equil_path, **molecule_config)

        elif relax:
            if use_deq:
                model_config_relax = dict(
                    model_tag=model_tag,
                    implicit_diff=True,
                    forw_tol=0,
                    back_tol=0,
                    forw_solver=fixed_point_iteration,
                    back_solver=fixed_point_iteration,
                    forw_max_iter=20,
                    back_max_iter=20,
                    warmstart_configs=dict(z=1, g_bar=1),
                )
            else:
                model_config_relax = dict(model_tag=model_tag)

            model, model_config_description_relaxed = load_update_desc_model(model=None, turn_into_deq=turn_into_deq, device=device, **model_config_relax)
            relax_path = proj_path('results-dev-relax' if dev else 'results', desc_molecule(molecule_name=molecule_name, molecule_idx=molecule_idx, model_config_relax=model_config_relax, fmax=0.01, stepsmax=1000, equil_steps=_equil_steps, equil_md_config=None, desc_equil=False), model_config_description_relaxed, 'relaxed_kcal_mol.xyz')
            equil_path = proj_path('results-dev-equil' if dev and (dev is True or 'equil' in dev) else 'results', desc_molecule(molecule_name=molecule_name, molecule_idx=molecule_idx, model_config_relax=model_config_relax, fmax=0.01, stepsmax=1000, equil_steps=_equil_steps, equil_md_config=None, desc_equil=True),  model_config_description_relaxed, 'equilibriated.pt')

            print(f'{relax_path=} Exists={os.path.exists(relax_path)}')
            print(f'{equil_path=} Exists={os.path.exists(equil_path)}')

            # --no-relax: skip this combination if relax result is missing
            if no_relax and not os.path.exists(relax_path):
                print(f'  Skip (--no-relax, missing): {relax_path}')
                skip_combination = True

            # --no-equil: if equil result missing, skip equil and MD for this combination
            if no_equil and equil_path and not os.path.exists(equil_path):
                print(f'  --no-equil: equil missing, disabling equil+MD for this combo')
                _equil_steps = 0
                _no_md = True

            equil_md_config = None
            if _equil_steps:
                equil_md_config = md_config.copy()
                equil_md_config['langevin_time_constant'] = langevin_time_constant_equilibration

            molecule_config = dict(
                molecule_name=molecule_name,
                molecule_idx=molecule_idx,
                model_config_relax=model_config_relax,
                fmax=0.01,
                stepsmax=1000,
                equil_steps=_equil_steps,
                equil_md_config=equil_md_config,
            )
            desc_molecule_str = desc_molecule(**molecule_config)

            def prepare_system(
                model=model, relax_path=relax_path, equil_path=equil_path,
                molecule_config=molecule_config, equil_md_config=equil_md_config,
                model_config_description_relaxed=model_config_description_relaxed,
            ):
                equil_hooks = None
                if _equil_steps and equil_path and not os.path.exists(equil_path) and equil_wandb:
                    equil_wandb_logger = WandBLoggerMD(
                        project='md-equilibration',
                        name='equil-',
                        tags=tags + ['equilibration'],
                        hyperparams=dict(
                            md_config=equil_md_config,
                            molecule_config=molecule_config,
                            model_config_description=model_config_description_relaxed if relax else '',
                            seed=seed,
                            phase='equilibration',
                        ),
                        log_properties=log_properties,
                        every_n_steps=1,
                    )
                    equil_hooks = [equil_wandb_logger]
                return load_relax_equil_atoms(model=model, relax_path=relax_path, equil_path=equil_path, equil_hooks=equil_hooks, **molecule_config)

        else:
            # No system preparation (no relax, no equilibrate).
            model = relax_path = equil_path = None
            model_config_relax = equil_md_config = None
            model_config_description_relaxed = ''
            molecule_config = dict(
                molecule_name=molecule_name,
                molecule_idx=molecule_idx,
                model_config_relax=model_config_relax,
                fmax=0.01,
                stepsmax=1000,
                equil_steps=_equil_steps,
                equil_md_config=equil_md_config,
            )
            desc_molecule_str = desc_molecule(**molecule_config)

            def prepare_system(model=model, relax_path=relax_path, equil_path=equil_path, molecule_config=molecule_config):
                return load_relax_equil_atoms(model=model, relax_path=relax_path, equil_path=equil_path, equil_hooks=None, **molecule_config)

        if skip_combination:
            continue

        log_properties = []
        if use_deq:
            log_properties = ['forw_iterations', 'back_iterations'] if implicit_diff else ['forw_iterations']

        if dev is not True:
            api = wandb.Api()
            training_config = api.artifact(model_tag).logged_by().config
            cfg = 'continue_from_config'
            if cfg in training_config:
                del training_config[cfg]
        else:
            training_config = None

        print(f'Running system: {molecule_name=}, {molecule_idx=}, {seed=}, {use_deq=}')

        run_md_tol_loop(
            prepare_system=prepare_system,
            desc_molecule_str=desc_molecule_str,
            model=model,
            model_tag=model_tag,
            molecule_name=molecule_name,
            molecule_config=molecule_config,
            md_config=md_config,
            seed=seed,
            use_deq=use_deq,
            tolerances_to_try=tolerances_to_try,
            no_md=_no_md,
            dev=dev,
            device=device,
            overwrite=overwrite,
            tol_grid_search=tol_grid_search,
            steps=steps,
            implicit_diff=implicit_diff,
            calc_setup=calc_setup,
            warmstart_mode=warmstart_mode,
            max_iter=max_iter,
            tags=tags,
            training_config=training_config,
            log_properties=log_properties,
            bonds_and_stats_per_molecule=bonds_and_stats_per_molecule,
        )
