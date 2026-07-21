import argparse
import os
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm

from schnetpack.md.data import HDF5Loader

from imlff.loading import load_model_from_wandb
from imlff.md.util import desc_md, setup_hdf5_logger, setup_md, load_update_desc_model, load_system_equilibrated
from imlff.common.util import proj_path
from imlff.model.solvers import fixed_point_iteration
from imlff.data.info import md17_conformations_per_molecule, md22_conformations_per_molecule
from imlff.md.stateful_extrapolator import StatefulAdamsBashforth

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from tqdm import tqdm
from filelock import FileLock

overwrite = False
MAX_WORKERS = 8 # int(os.environ.get("N_WORKERS", os.cpu_count() or 1))

focus = 'temperature'
focus = 'regularization'
focus = 'warmstarts'

overwrite = False

direction = 1
# direction = -1
    
temperatures = [500, 300]
ws_degrees   = [-1]
tolerances   = [0, 0.001, 0.01]
seeds = list(range(10))
md_config = dict(timestep=0.5,)

if focus == 'regularization':
    runs_df = pd.read_csv(proj_path('result_summary/runs/april-22_aspirin_param_search/runs_df_val_test.csv')).query('type=="deq"')
    molecule_and_model = runs_df[['molecule', 'best_manual_tag']].to_numpy()    
    temperatures = [300]
    ws_degrees   = [-1]
    # ws_degrees   = [0, 1]
    tolerances   = [0.01]
elif 'warmstarts' in focus:
    molecule_and_model = [('aspirin', 'qcml-group/dev-v10/model-6nf8mqsm:v579')] # jac_reg 0.32 (from 'grid search')
    tolerances   = [0.01]
    ws_degrees   = [-1, 0, 1, 2, 3, 4]
    
    # MAX_WORKERS = 4
    # temperatures = [300]
elif 'temperature' in focus:
    temperatures = [100, 200, 300, 400, 500, 600]  
    molecule_and_model = [('Ac-Ala3-NHMe', 'qcml-group/dev-v10/model-4ehmqw9w:v3392')] # jac_reg 0.32

print('Running models:', molecule_and_model)
    
print(f'{temperatures=}')
print(f'{tolerances=}')
print(f'{ws_degrees=}')
print(f'{seeds=}')
print(f'{molecule_and_model=}')


try:
    # fdefined in IPython/Jupyter, raises NameError otherwise:
    get_ipython()  
    array_idx = array_stride = reverse = None
except NameError:
    parser = argparse.ArgumentParser()
    parser.add_argument('array_idx', type=int, nargs='?', default=None)
    parser.add_argument('array_stride', type=int, nargs='?', default=None, help="Stride for array job splitting.")
    parser.add_argument('--reverse', action='store_true', help="Stride for array job splitting.")
    args = parser.parse_args()
    array_idx = args.array_idx
    array_stride = args.array_stride
    reverse = args.reverse

if array_idx is not None:
    if array_stride is not None:
        molecule_and_model = molecule_and_model[array_idx::array_stride]
    else:
        molecule_and_model = molecule_and_model[array_idx:array_idx+1]
    print('After array selection:')
    print(f'{molecule_and_model=}')

if reverse:
    direction *= -1
setups = list(product(
    molecule_and_model,
    temperatures,
    tolerances,
    ws_degrees,
    seeds,
))[::direction]
print(len(setups))
# exit()


def run_setup(args):
    (molecule_name, model_tag), target_temperature, tol, warmstart_degree, seed = args
    tol = float(tol)

    prefix = 'qcml-group/dev-v10/'
    if not model_tag.startswith(prefix):
        model_tag = prefix + model_tag
    model = load_model_from_wandb(model_tag)

    print(f"Running seed {seed} with warmstart degree {warmstart_degree}")

    model_config = dict(
        model_tag=model_tag,
        implicit_diff=True,
        forw_tol=tol,
        back_tol=tol,
        forw_solver=fixed_point_iteration,
        back_solver=fixed_point_iteration,
        forw_max_iter=10,
        back_max_iter=10,
        warmstart_configs=dict(z=warmstart_degree, g_bar=warmstart_degree) if warmstart_degree >= 0 else None,
    )
    model, model_config_description = load_update_desc_model(model=model, **model_config)

    np.random.seed(seed)
    n_conformations = md17_conformations_per_molecule.get(molecule_name, md22_conformations_per_molecule.get(molecule_name))
    assert n_conformations is not None, f"Unknown molecule: {molecule_name}"
    molecule_config = dict(
        molecule_name = molecule_name,
        molecule_idx = np.random.randint(0, n_conformations),
        equilibrate_strategy = 1,
        initial_temperature = 500,
        target_temperature = target_temperature,
    )
    
    setup_path = proj_path('results', f"iter_count/s{molecule_config['equilibrate_strategy']}/{molecule_config['molecule_name']}/T{molecule_config['target_temperature']:04d}K_iT{molecule_config['initial_temperature']:04d}K/{molecule_config['molecule_idx']}/")
    equil_path = setup_path + f"equilibrated_{model_tag.split('model-')[1]}.pt"
    equil_lock = FileLock(equil_path + ".lock")
    with equil_lock:
        system = load_system_equilibrated(model=model, equil_path=equil_path, **molecule_config)

    md_log_path = setup_path + desc_md(**md_config) + '__' + model_config_description
    log_properties = [f'{s}_res_{i:02d}' for s in ['forw', 'back'] for i in range(1, 11)]

    # lock around md_log_path
    log_lock = FileLock(md_log_path + ".lock")
    with log_lock:
        if overwrite==False and os.path.exists(md_log_path + '.hdf5'):
            with HDF5Loader(f"{md_log_path}.hdf5", skip_initial=0) as data:
                l = len(data.get_property(log_properties[0], atomistic=False))
                print('\n' * 10, l)
                if l == 1000:
                    print('found:', args)
                    return  # already done

        logger = setup_hdf5_logger(md_log_path, overwrite='yes', log_properties=log_properties)

        md_simulator, _ = setup_md(
            model, system, md_config,
            warmstart_configs=model_config.get('warmstart_configs', dict()),
            StatefulClass=StatefulAdamsBashforth,
            hooks=[logger],
        )
        md_simulator.progress = False
        md_simulator.simulate(1000)

if 1:
    # ---- run in parallel ----
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(run_setup, setup) for setup in setups]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Setups"):
            pass
else:
    for s in setups[-1:]:
        run_setup(s)
        
