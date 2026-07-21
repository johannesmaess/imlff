import torch
import random
import os
import numpy as np
import math
import gc
from tqdm import tqdm
from functools import lru_cache

from schnetpack.representation import SchNet, PaiNN, SO3net

from imlff.model.util import ForcesAndDeriveGBar
from imlff.model.solvers import fixed_point_iteration, gather_max_within_molecule
from imlff.model.linearization import fw_linearize_spk
from imlff.md.util import short_name, warmstart_config_str
from imlff.loading import artifact_or_run_cached, load_model_from_wandb
from imlff.data.util import load_md17_data
from imlff.common.util import proj_path



def _extract_successor_batch(batch, successor=1):
    inputs = dict()
    changed_keys = ['_idx', 'energy', 'forces', '_n_atoms', '_atomic_numbers', '_positions', '_cell', '_pbc']
    shared_keys = ['_offsets', '_idx_m', '_idx_i', '_idx_j']
    postfix = f'_{successor:03d}' if successor>0 else ''
    for k in changed_keys:
        inputs[k] = batch[k+postfix]
    for k in shared_keys:
        inputs[k] = batch[k]
    return inputs


def _all_exist(paths):
    return all(os.path.exists(p) for p in paths)


def _load_path_checked(path, *, device=None):
    """Load a .npy or .pt file and sanity-check that it contains no NaNs."""
    if path.endswith(".npy"):
        arr = np.load(path)
        if np.any(np.isnan(arr)):
            raise ValueError(f"{path} has nan entries")
        return arr
    if path.endswith(".pt"):
        dev = torch.device(device) if device is not None else None
        t = torch.load(path, map_location=dev, weights_only=True)
        if torch.any(torch.isnan(t)):
            raise ValueError(f"{path} has nan entries")
        return t
    raise ValueError(f"Unsupported file type for {path}")


def _try_load_key(dir_path: str, key: str, *, device=None):
    """Try loading `<dir_path>/<key>.npy` or `.pt` (in that order). Return None if missing."""
    for ext in (".npy", ".pt"):
        path = f"{dir_path}{key}{ext}"
        if os.path.exists(path):
            return _load_path_checked(path, device=device)
    return None

def clear_warmstart_run_caches(empty_cuda_cache: bool = True, gc_collect: bool = True) -> None:
    try:
        try:
            loader_cached.cache_clear()
        except:
            pass
        # Keep warm-iteration cache in sync with callers that clear caches on OOM retries.
        clear_extrapolated_init_cache()
    finally:
        if gc_collect:
            gc.collect()
        if empty_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()


@lru_cache(maxsize=2)
def _get_extrapolated_init_cached(default_path: str, kind: str, degree: int):
    """
    Construct an extrapolated init tensor from saved *_star_{prev,curr}.pt.

    degree=0: constant (curr)
    degree=1: linear (2*curr - prev)

    The cached tensor is kept on CPU — it is full-sized (n_conformations, *z_shape) and
    can be tens of GB for SO3net/large molecules. Callers slice a per-batch chunk and move
    only that chunk to the GPU (see get_extrapolated_init call sites).

    Cache is intentionally tiny (2 entries) because callers typically need only two
    variants (e.g. z and g_bar) across tight warm-iter loops.
    """
    if kind not in {"z", "g", "g_bar"}:
        raise ValueError(f"Unknown {kind=}. Expected one of: z, g, g_bar.")
    if degree not in (0, 1):
        raise ValueError(f"Unsupported {degree=}. Expected 0 or 1.")

    curr_path = f"{default_path}{kind}_star_curr.pt"
    curr = _load_path_checked(curr_path, device="cpu")
    if degree == 0:
        return curr

    prev_path = f"{default_path}{kind}_star_prev.pt"
    prev = _load_path_checked(prev_path, device="cpu")
    # in-place to avoid a third full-sized tensor
    curr.mul_(2).sub_(prev)
    del prev
    return curr


def get_extrapolated_init(default_path: str, kind: str, degree: int, device=None):
    # The full tensor stays cached on CPU; `device` is accepted for signature stability
    # but slicing+to(device) of the per-batch chunk happens at the call site.
    return _get_extrapolated_init_cached(default_path, kind, degree)


def clear_extrapolated_init_cache() -> None:
    _get_extrapolated_init_cached.cache_clear()


@lru_cache(maxsize=1)
def loader_cached(partition='val', limit=None, **data_config):
    module = load_md17_data(**data_config, duplicate_for_run=True)
    
    if type(partition) is str:
        if partition == 'train':
            idx = module.train_idx
        elif partition == 'val':
            idx = module.val_idx
        elif partition == 'test':
            idx = module.test_idx
        else:
            raise ValueError(f'Unknown partition: {partition}')
           
        # start with two predecessors (note that they are not guaranteed to be in the dataset partition)
        idx = sorted([i-2 for i in idx if i>1])
        random.seed(1)  # Set seed for reproducibility
        random.shuffle(idx)  # Shuffle the indices pseudo-randomly
        if limit: idx = idx[:limit]
        
    elif type(partition) is tuple:
        idx = partition
    else:
        raise ValueError(f'Unknown partition: {partition}')
     
    # overwrite test dataset with requested subset. this assumes that transforms are the same.
    module.test_dataset.subset_idx = idx
    return module.test_dataloader()


def _get_rep(nnpot, is_deq=True):
    """Extract the AtomisticRepresentationStackable from an nnpot model."""
    if is_deq:
        return nnpot.representation.f
    return nnpot.representation.stack_schnets[0]


def get_z_shape(nnpot, n_atoms, is_deq=True):
    """Get the per-atom z shape tuple from an nnpot model.

    Returns a single tuple, e.g. (n_atoms, 128) for SchNet, (n_atoms, 4, 128) for PaiNN.
    """
    rep = _get_rep(nnpot, is_deq)
    n_z = rep.n_z
    rep = rep.AtomisticRepresentationClass.__name__

    if rep == 'PaiNN':
        return (n_atoms, 4, n_z)
    if rep == 'SO3net':
        return (n_atoms, (rep.lmax + 1) ** 2, n_z)
    # SchNet or unknown
    return (n_atoms, n_z)


def get_degree_info(nnpot, is_deq=True):
    """Get degree split info for per-degree residual analysis.

    Returns (degree_splits, basis_dim, has_degree_dim):
      - degree_splits: tuple of ints summing to basis_dim
      - basis_dim: total size of the basis (degree) dimension
      - has_degree_dim: whether the z tensor has a separate basis axis
    """
    rep = _get_rep(nnpot, is_deq)
    Rep = rep.AtomisticRepresentationClass

    if Rep is PaiNN:
        return (1, 3), 4, True
    if Rep is SO3net:
        splits = tuple(2 * l + 1 for l in range(rep.lmax + 1))
        return splits, sum(splits), True
    
    assert Rep is SchNet
    # SchNet: no separate basis axis
    return (1,), 1, False


def analyze_warmstart(
    *args,
    bsz=1,
    per_batch=False,
    n_bs=1_000_000, # larger than any md17/md22 dataset
    aggregate=True,
    limit=20000,
    **kwargs,
):
    
    # get model tag from run config, 
    # molecule = analyze_warmstart_core(*args, return_molecule=True, **kwargs)
    # n_available = molecule_to_n_conformations(molecule)
    # if limit is None or limit > n_available:
    #     limit = n_available
    if bsz is not None:
        limit = min(limit, n_bs * bsz)
        n_bs = math.ceil(limit / bsz)
    else:
        n_bs = limit
    kwargs.setdefault('return_key', 'F_mae_next')
    
    if not per_batch:
        return analyze_warmstart_core(*args, n_bs=n_bs, limit=limit, bsz=bsz, **kwargs)
    
    if aggregate:
        assert 'return_key' in kwargs, 'aggregate requires return_key to be set'
        assert 'return_paths' not in kwargs, 'aggregate does not support return_paths'
        
        # target path for aggregated result
        results_path, default_path, warm_path = analyze_warmstart_core(*args, return_paths=True, limit=limit, **kwargs) 
        path_with_key = f'{warm_path}{kwargs["return_key"]}'
        
        try:
            return np.load(f'{path_with_key}.npy')
        except FileNotFoundError:
            pass
        try:
            return torch.load(f'{path_with_key}.pt', map_location=kwargs.get('device'))
        except FileNotFoundError:
            pass
        
    batch_results = []
    for i in range(n_bs):
        print(f'Analyzing batch {i}/{n_bs}...')
        result = analyze_warmstart_core(*args, only_batch=i, limit=limit, bsz=bsz, **kwargs)
        if type(result) is str and result == 'surpassed n_batches available':
            break
        assert result is not None, f'Batch {i} returned None, but should return a result.'
        if aggregate:
            batch_results.append(result)
            
    if aggregate:
        if len(batch_results) == 0:
            print('No results found for any batch.')
            return None
        
        print('Saving aggregated results under:\n', warm_path)
        os.makedirs(os.path.dirname(warm_path), exist_ok=True)
        
        if isinstance(batch_results[0], np.ndarray):
            full = np.concatenate(batch_results, axis=0)
            if kwargs['save']:
                np.save(f'{path_with_key}.npy', full)
        elif isinstance(batch_results[0], torch.Tensor):
            full = torch.cat(batch_results, dim=0)
            if kwargs['save']:
                torch.save(full, f'{path_with_key}.pt')
        
        return full
    
def _warmstart_paths(
    *,
    molecule,
    run_name,
    model_tag,
    is_deq,
    partition,
    limit,
    only_batch,
    bsz,
    default_forw_tol,
    default_back_tol,
    default_max_iter,
    back_iter_on_g_bar,
    z, g, g_bar,
    implicit_diff,
    forw_max_iter,
    back_max_iter,
    dh_dR_iter,
    results_dir,
    use_tmp,
):
    """Return (results_path, default_path, warm_path) without any I/O or model loading."""
    version = '7' if is_deq else '6'
    model_str = f'{short_name(run_name)}_{model_tag.replace("/", "_")}'
    results_path = proj_path(f'{results_dir}/{molecule}/{model_str}/warmstart_analysis_V' + version, use_tmp=use_tmp)
    if partition != 'val':
        results_path += f'_{partition}'
    if only_batch is not None:
        results_path += f'/bsz{bsz:03d}_batch{only_batch:04d}'
    elif limit:
        results_path += f'_{limit:06d}'
    results_path += '/'

    if is_deq:
        default_config_str = ''
        if default_forw_tol >= 0:
            default_config_str += f'tol{default_forw_tol}_'
        default_config_str += f'mi{default_max_iter:02d}'
        if default_forw_tol != default_back_tol >= 0:
            default_config_str += f'_btol{default_back_tol}'
        if back_iter_on_g_bar:
            default_config_str += '_on_g_bar'
        warm_config_str = warmstart_config_str(z=z, g_bar=g_bar, g=g)
        if is_deq:
            if implicit_diff:
                warm_config_str += f'_mi{forw_max_iter:02d}'
                if forw_max_iter != back_max_iter:
                    warm_config_str += f'_bmi{back_max_iter:02d}'
                if back_iter_on_g_bar:
                    warm_config_str += '_iter_g_bar'
            else:
                # No implicit diff
                assert g < 0 and g_bar < 0, 'Only z warmstart supported without implicit diff'
                warm_config_str += '_FNI{forw_max_iter:02d}'
            if dh_dR_iter:
                warm_config_str += f'_dh_dR{dh_dR_iter:02d}'
                assert implicit_diff and z == 1, 'dh/dR only supported with z warmstart'
        default_path = f'{results_path}{default_config_str}/'
        if z >= 0 or g >= 0 or g_bar >= 0:
            # it matters from what we warmstarted (default tol, btol, mi, bmi)
            warm_path = f'{default_path}{warm_config_str}/'
        else:
            # we did not warmstart
            warm_path = f'{results_path}{warm_config_str}/'
    else:
        warm_path = default_path = results_path

    return results_path, default_path, warm_path


def warmstart_exists(
    model_tag,
    config,
    forw_max_iter,
    back_max_iter,
    *,
    default_forw_tol=-1,
    default_back_tol=-1,
    default_max_iter=10,
    back_iter_on_g_bar=False,
    z=1,
    g=-1,
    g_bar=1,
    implicit_diff=True,
    dh_dR_iter=0,
    partition='test',
    limit=None,
    results_dir='results',
    use_tmp=False,
    turn_into_deq=False,
):
    """Return True iff F_mae_next.npy exists on disk for this (model, forw_mi, back_mi) combination."""
    is_deq = 'mi' in config['model_key'] or turn_into_deq
    run_name = config.get('name')
    if run_name is None:
        run_tag = model_tag.split('model-')[-1].split(':')[0]
        run_name = artifact_or_run_cached(run_tag, run=True).name
    _, _, warm_path = _warmstart_paths(
        molecule=config['molecule'],
        run_name=run_name,
        model_tag=model_tag,
        is_deq=is_deq,
        partition=partition,
        limit=limit,
        only_batch=None,
        bsz=None,
        default_forw_tol=default_forw_tol,
        default_back_tol=default_back_tol,
        default_max_iter=default_max_iter,
        back_iter_on_g_bar=back_iter_on_g_bar,
        z=z, g=g, g_bar=g_bar,
        implicit_diff=implicit_diff,
        forw_max_iter=forw_max_iter,
        back_max_iter=back_max_iter,
        dh_dR_iter=dh_dR_iter,
        results_dir=results_dir,
        use_tmp=use_tmp,
    )
    return os.path.exists(f'{warm_path}F_mae_next.npy')


def analyze_warmstart_core(
    model_tag,
    config=None,

    default_max_iter = 10,
    default_forw_tol = -1,
    default_back_tol = -1,
    forw_max_iter = 10,
    back_max_iter = 10,
    dh_dR_iter = 0,
    back_iter_on_g_bar=False,
    
    implicit_diff = True,
    # warmstart_configs = dict(z=1, g_bar=1),
    z=1,
    g=-1,
    g_bar=1,

    bsz = 10,
    n_bs = None,
    only_batch = None,
    
    partition = 'test',
    limit = None,
        
    measure_default = 'only_next',
    measure_restricted = 'auto',
    
    save='star',
    save_preds=True,
    save_atomwise_errors=False,
    save_atomwise_residuals=False,
    save_atomwise_degree_residuals=False,
    save_star_restricted=False,
    return_paths = False,
    results_dir = 'results',
    
    turn_into_deq = False,
    device = None,
    
    return_key='F_mae_next',
    return_warm=True,
    
    use_tmp=False,
):
    assert z in [-1, 0, 1]
    assert g in [-1, 0, 1]
    assert g_bar in [-1, 0, 1]
    if type(partition) is not str:
        assert type(partition) is tuple
        assert not save
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')
    
    ### fetch
    # Each row in our selection dataframes already contains the fields stored in `run.config`.
    # Allow passing them through to avoid extra wandb/API calls.
    run = None
    run_tag = model_tag.split('model-')[-1].split(':')[0]
    if config is None:
        run = artifact_or_run_cached(run_tag, run=True)
        config = run.config
    
    data_config = dict(
                num_train=config['num_train'],
                num_val=config['num_val'],
                molecule=config['molecule'],
                load_successors=2,
                batch_size=bsz,
                split_id=config['split_id'],
                adj_cutoff = config.get('cutoff', np.inf) if config['Dataset']=='MD22' else np.inf,
                num_workers=1 if only_batch else 4,
            )
    
    is_deq = 'mi' in config['model_key'] or turn_into_deq

    if n_bs:
        limit_through_batches = bsz * n_bs
        if limit is None or limit > limit_through_batches:
            limit = limit_through_batches

    # version = '_codex_test'
    run_name = config.get('name')
    if run_name is None:
        # Fallback: load run just for its name (needed for stable on-disk paths).
        run = run or artifact_or_run_cached(run_tag, run=True)
        run_name = run.name

    results_path, default_path, warm_path = _warmstart_paths(
        molecule=data_config['molecule'],
        run_name=run_name,
        model_tag=model_tag,
        is_deq=is_deq,
        partition=partition,
        limit=limit,
        only_batch=only_batch,
        bsz=bsz,
        default_forw_tol=default_forw_tol,
        default_back_tol=default_back_tol,
        default_max_iter=default_max_iter,
        back_iter_on_g_bar=back_iter_on_g_bar,
        z=z, g=g, g_bar=g_bar,
        implicit_diff=implicit_diff,
        forw_max_iter=forw_max_iter,
        back_max_iter=back_max_iter,
        dh_dR_iter=dh_dR_iter,
        results_dir=results_dir,
        use_tmp=use_tmp,
    )

    if return_paths:
        return results_path, default_path, warm_path
    
    # create dirs
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(default_path, exist_ok=True)
    os.makedirs(warm_path, exist_ok=True)

    if return_key and (measure_restricted is not True) and (measure_default is not True):
        load_from = default_path
        if is_deq and return_warm and measure_restricted:
            load_from = warm_path
        a = _try_load_key(load_from, return_key, device=device)
        if a is not None:
            return a
        
    for p in [warm_path, default_path, results_path]:
        os.makedirs(p, exist_ok=True)
    
    def get_nnpot():
        ### load and configure
        nnpot = load_model_from_wandb(model_tag, turn_into_deq=turn_into_deq).to(device)
        nnpot.eval()
        deq_obj = None
        if is_deq:
            deq_obj = nnpot.representation.deq
            assert any(isinstance(om, ForcesAndDeriveGBar) for om in nnpot.output_modules)
            
            deq_obj.implicit_diff = implicit_diff
            deq_obj.forw_solver = deq_obj.back_solver = fixed_point_iteration
            deq_obj.forw_max_iter = deq_obj.back_max_iter = default_max_iter
            deq_obj.forw_tol = deq_obj.back_tol = -1
            deq_obj.iterate_on_g_bar = back_iter_on_g_bar
            
        return nnpot, deq_obj
    nnpot = deq_obj = None
    
    ### prepare logging
    os.makedirs(results_path, exist_ok=True)
    print(f'Saving results to:')
    print(results_path)
    print(default_path)
    print(warm_path)
    # exit()
    
    def run_default_pass():
        iterator = val_loader = loader_cached(partition=partition, limit=limit, **data_config)
        n_atoms = next(iter(val_loader))['_n_atoms'].detach()[0]
        n_batches = n_bs or len(val_loader)
        
        if only_batch is not None:
            if only_batch >= n_batches:
                print(f'Only batch {only_batch} requested, but only {n_batches} batches available. Exiting.')
                return 'surpassed n_batches available'
            else:
                print(f'{n_batches} batches available, but only batch {only_batch} requested. Running only this batch.')
                n_batches = 1
            
            # extract the only batch that we want to process
            org = val_loader.dataset.subset_idx.copy()
            try:
                val_loader.dataset.subset_idx = org[only_batch*bsz:(only_batch+1)*bsz]
                iterator = [next(iter(val_loader))]
            finally:
                val_loader.dataset.subset_idx = org
        
        if True: ### stats and conversions
            nnpot, deq_obj = get_nnpot()
            
            n_conformations = min(len(val_loader.dataset), bsz * n_batches)
            z_shape = get_z_shape(nnpot, n_atoms, is_deq)
            n_atoms_int = int(n_atoms)
            z_shape_batched = (bsz*n_atoms, *z_shape[1:])
            process = lambda z: z.detach().clone().reshape(-1, *z_shape)
            # The full (n_conformations, *z_shape) star accumulators are held on CPU
            # (they can be tens of GB for SO3net/large molecules); only per-batch slices
            # ever live on GPU. process_cpu lands a batch's fixpoint on CPU for accumulation.
            process_cpu = lambda z: z.detach().reshape(-1, *z_shape).cpu()
            mean_absolute_error = lambda a, b, bs=bsz: torch.mean(torch.abs(a.view(bs, -1)-b.view(bs, -1)), dim=1).detach().cpu().numpy()
            mean_absolute_error_per_atom = lambda a, b, bs=bsz: torch.mean(
                torch.abs(a.view(bs, n_atoms_int, -1) - b.view(bs, n_atoms_int, -1)),
                dim=2,
            ).detach().cpu().numpy()
            forces_to_numpy = lambda f, bs=bsz: f.view(bs, n_atoms_int, 3).detach().cpu().numpy()
            energy_to_numpy = lambda e, bs=bsz: e.view(bs).detach().cpu().numpy()
            res_to_numpy = lambda r, bs=bsz: r.view(bs, n_atoms_int).detach().cpu().numpy()
        
        if only_batch is None: ### inits, saving
            if is_deq or ('only_next' != measure_default):
                F_mae_prev = np.full(n_conformations, np.nan)
                E_mae_prev = np.full(n_conformations, np.nan)
                if save_atomwise_errors:
                    F_mae_per_atom_prev = np.full((n_conformations, n_atoms_int), np.nan)
                if save_preds:
                    F_pred_prev = np.full((n_conformations, n_atoms_int, 3), np.nan)
                    E_pred_prev = np.full(n_conformations, np.nan)

                F_mae_curr = np.full(n_conformations, np.nan)
                E_mae_curr = np.full(n_conformations, np.nan)
                if save_atomwise_errors:
                    F_mae_per_atom_curr = np.full((n_conformations, n_atoms_int), np.nan)
                if save_preds:
                    F_pred_curr = np.full((n_conformations, n_atoms_int, 3), np.nan)
                    E_pred_curr = np.full(n_conformations, np.nan)

                if is_deq and save_atomwise_residuals:
                    forw_res_prev_per_atom = np.full((n_conformations, n_atoms_int), np.nan)
                    back_res_prev_per_atom = np.full((n_conformations, n_atoms_int), np.nan)
                    forw_res_curr_per_atom = np.full((n_conformations, n_atoms_int), np.nan)
                    back_res_curr_per_atom = np.full((n_conformations, n_atoms_int), np.nan)
            
            if is_deq or measure_default:
                E_mae_next = np.full(n_conformations, np.nan)
                F_mae_next = np.full(n_conformations, np.nan)
                if save_atomwise_errors:
                    F_mae_per_atom_next = np.full((n_conformations, n_atoms_int), np.nan)
                if save_preds:
                    F_pred_next = np.full((n_conformations, n_atoms_int, 3), np.nan)
                    E_pred_next = np.full(n_conformations, np.nan)
                if is_deq and save_atomwise_residuals:
                    forw_res_next_per_atom = np.full((n_conformations, n_atoms_int), np.nan)
                    back_res_next_per_atom = np.full((n_conformations, n_atoms_int), np.nan)

            # Accumulators live on CPU; see process_cpu above.
            if is_deq or ('only_next' != measure_default):
                z_star_prev     = torch.full((n_conformations, *z_shape), torch.nan, device='cpu')
                g_star_prev     = torch.full((n_conformations, *z_shape), torch.nan, device='cpu')
                g_bar_star_prev = torch.full((n_conformations, *z_shape), torch.nan, device='cpu')

                z_star_curr     = torch.full((n_conformations, *z_shape), torch.nan, device='cpu')
                g_star_curr     = torch.full((n_conformations, *z_shape), torch.nan, device='cpu')
                g_bar_star_curr = torch.full((n_conformations, *z_shape), torch.nan, device='cpu')

            if is_deq or measure_default:
                z_star_next     = torch.full((n_conformations, *z_shape), torch.nan, device='cpu')
                g_star_next     = torch.full((n_conformations, *z_shape), torch.nan, device='cpu')
                g_bar_star_next = torch.full((n_conformations, *z_shape), torch.nan, device='cpu')

        else:
            g_star_prev = g_star_curr = g_star_next = None

        ### compute
        for j, batch in enumerate(tqdm(iterator)): ### run and save
            bs = len(batch['_n_atoms'])
            if j>=n_batches:
                break
            
            batch = {key: value.to(device) for key, value in batch.items()}
        
            ### first two forward passes: many iterations, obtain fix points and predictions
        
            if is_deq or ('only_next' != measure_default):
                ## 'prev' - for having fixpoints at t-2 in the cold/warmstart tests
                inputs = _extract_successor_batch(batch, successor=0)
                F_prev = inputs['forces'].clone()
                E_prev = inputs['energy'].clone()
                inputs['forw_tol'] = default_forw_tol
                inputs['back_tol'] = default_back_tol
                out = nnpot(inputs)
                prev_forces = out['forces']
                prev_energy = out['energy']
                
                if only_batch is None:
                    F_mae_prev[j*bsz:j*bsz+bs] = mean_absolute_error(prev_forces, F_prev, bs=bs)
                    E_mae_prev[j*bsz:j*bsz+bs] = mean_absolute_error(prev_energy, E_prev, bs=bs)
                    if save_atomwise_errors:
                        F_mae_per_atom_prev[j*bsz:j*bsz+bs] = mean_absolute_error_per_atom(prev_forces, F_prev, bs=bs)
                    if save_preds:
                        F_pred_prev[j*bsz:j*bsz+bs] = forces_to_numpy(prev_forces, bs=bs)
                        E_pred_prev[j*bsz:j*bsz+bs] = energy_to_numpy(prev_energy, bs=bs)
                    if is_deq and save_atomwise_residuals:
                        forw_res_prev_per_atom[j*bsz:j*bsz+bs] = res_to_numpy(out['forw_res'][max(out['forw_res'])], bs=bs)
                        back_res_prev_per_atom[j*bsz:j*bsz+bs] = res_to_numpy(out['back_res'][max(out['back_res'])], bs=bs)
                    
                    if 'z_star' in out:
                        z_star_prev[j*bsz:j*bsz+bs] =     process_cpu(out['z_star'])
                    if 'g_star' in out:
                        g_star_prev[j*bsz:j*bsz+bs] =     process_cpu(out['g_star'])
                    if 'g_bar_star' in out:
                        g_bar_star_prev[j*bsz:j*bsz+bs] = process_cpu(out['g_bar_star'])
                else:
                    F_mae_prev = mean_absolute_error(prev_forces, F_prev, bs=bs)
                    E_mae_prev = mean_absolute_error(prev_energy, E_prev, bs=bs)
                    if save_atomwise_errors:
                        F_mae_per_atom_prev = mean_absolute_error_per_atom(prev_forces, F_prev, bs=bs)
                    if save_preds:
                        F_pred_prev = forces_to_numpy(prev_forces, bs=bs)
                        E_pred_prev = energy_to_numpy(prev_energy, bs=bs)
                    if is_deq and save_atomwise_residuals:
                        forw_res_prev_per_atom = res_to_numpy(out['forw_res'][max(out['forw_res'])], bs=bs)
                        back_res_prev_per_atom = res_to_numpy(out['back_res'][max(out['back_res'])], bs=bs)
                    
                    if 'z_star' in out: 
                        z_star_prev =     process(out['z_star'])
                    if 'g_star' in out: 
                        g_star_prev =     process(out['g_star'])
                    if 'g_bar_star' in out: 
                        g_bar_star_prev = process(out['g_bar_star'])
                
                ## 'curr' - for having fixpoints at t-1 in the cold/warmstart tests
                inputs = _extract_successor_batch(batch, successor=1)
                F_curr = inputs['forces'].clone()
                E_curr = inputs['energy'].clone()
                R_curr = inputs['_positions']
                inputs['forw_tol'] = default_forw_tol
                inputs['back_tol'] = default_back_tol
                out = nnpot(inputs)
                curr_forces = out['forces']
                curr_energy = out['energy']
                
                if only_batch is None:
                    F_mae_curr[j*bsz:j*bsz+bs] = mean_absolute_error(curr_forces, F_curr, bs=bs)
                    E_mae_curr[j*bsz:j*bsz+bs] = mean_absolute_error(curr_energy, E_curr, bs=bs)
                    if save_atomwise_errors:
                        F_mae_per_atom_curr[j*bsz:j*bsz+bs] = mean_absolute_error_per_atom(curr_forces, F_curr, bs=bs)
                    if save_preds:
                        F_pred_curr[j*bsz:j*bsz+bs] = forces_to_numpy(curr_forces, bs=bs)
                        E_pred_curr[j*bsz:j*bsz+bs] = energy_to_numpy(curr_energy, bs=bs)
                    if is_deq and save_atomwise_residuals:
                        forw_res_curr_per_atom[j*bsz:j*bsz+bs] = res_to_numpy(out['forw_res'][max(out['forw_res'])], bs=bs)
                        back_res_curr_per_atom[j*bsz:j*bsz+bs] = res_to_numpy(out['back_res'][max(out['back_res'])], bs=bs)
                    if 'z_star' in out:
                        z_star_curr[j*bsz:j*bsz+bs]     = process_cpu(out['z_star'])
                    if 'g_star' in out:
                        g_star_curr[j*bsz:j*bsz+bs]     = process_cpu(out['g_star'])
                    if 'g_bar_star' in out:
                        g_bar_star_curr[j*bsz:j*bsz+bs] = process_cpu(out['g_bar_star'])
                else:
                    F_mae_curr = mean_absolute_error(curr_forces, F_curr, bs=bs)
                    E_mae_curr = mean_absolute_error(curr_energy, E_curr, bs=bs)
                    if save_atomwise_errors:
                        F_mae_per_atom_curr = mean_absolute_error_per_atom(curr_forces, F_curr, bs=bs)
                    if save_preds:
                        F_pred_curr = forces_to_numpy(curr_forces, bs=bs)
                        E_pred_curr = energy_to_numpy(curr_energy, bs=bs)
                    if is_deq and save_atomwise_residuals:
                        forw_res_curr_per_atom = res_to_numpy(out['forw_res'][max(out['forw_res'])], bs=bs)
                        back_res_curr_per_atom = res_to_numpy(out['back_res'][max(out['back_res'])], bs=bs)
                    if 'z_star' in out: 
                        z_star_curr     = process(out['z_star'])
                    if 'g_star' in out: 
                        g_star_curr     = process(out['g_star'])
                    if 'g_bar_star' in out: 
                        g_bar_star_curr = process(out['g_bar_star'])
        
            if is_deq or measure_default:
                ### 'next' - test different warmstarting strategies and budgets
                inputs = _extract_successor_batch(batch, successor=2)
                
                F_next = inputs['forces'].clone()
                E_next = inputs['energy'].clone()
                
                # calculate true fixed point: what is the ideal we can approximate with warmstarts?
                inputs['forw_tol'] = default_forw_tol
                inputs['back_tol'] = default_back_tol
                out = nnpot(inputs)
                next_forces = out['forces']
                next_energy = out['energy']
                
                if only_batch is None:
                    F_mae_next[j*bsz:j*bsz+bs] = mean_absolute_error(next_forces, F_next, bs=bs)
                    E_mae_next[j*bsz:j*bsz+bs] = mean_absolute_error(next_energy, E_next, bs=bs)
                    if save_atomwise_errors:
                        F_mae_per_atom_next[j*bsz:j*bsz+bs] = mean_absolute_error_per_atom(next_forces, F_next, bs=bs)
                    if save_preds:
                        F_pred_next[j*bsz:j*bsz+bs] = forces_to_numpy(next_forces, bs=bs)
                        E_pred_next[j*bsz:j*bsz+bs] = energy_to_numpy(next_energy, bs=bs)
                    if is_deq and save_atomwise_residuals:
                        forw_res_next_per_atom[j*bsz:j*bsz+bs] = res_to_numpy(out['forw_res'][max(out['forw_res'])], bs=bs)
                        back_res_next_per_atom[j*bsz:j*bsz+bs] = res_to_numpy(out['back_res'][max(out['back_res'])], bs=bs)
                
                    if 'z_star' in out:
                        z_star_next[j*bsz:j*bsz+bs]     = process_cpu(out['z_star'])
                    if 'g_star' in out:
                        g_star_next[j*bsz:j*bsz+bs]     = process_cpu(out['g_star'])
                    if 'g_bar_star' in out:
                        g_bar_star_next[j*bsz:j*bsz+bs] = process_cpu(out['g_bar_star'])
                else:
                    F_mae_next = mean_absolute_error(next_forces, F_next, bs=bs)
                    E_mae_next = mean_absolute_error(next_energy, E_next, bs=bs)
                    if save_atomwise_errors:
                        F_mae_per_atom_next = mean_absolute_error_per_atom(next_forces, F_next, bs=bs)
                    if save_preds:
                        F_pred_next = forces_to_numpy(next_forces, bs=bs)
                        E_pred_next = energy_to_numpy(next_energy, bs=bs)
                    if is_deq and save_atomwise_residuals:
                        forw_res_next_per_atom = res_to_numpy(out['forw_res'][max(out['forw_res'])], bs=bs)
                        back_res_next_per_atom = res_to_numpy(out['back_res'][max(out['back_res'])], bs=bs)
                    
                    if 'z_star' in out: 
                        z_star_next     = process(out['z_star'])
                    if 'g_star' in out: 
                        g_star_next     = process(out['g_star'])
                    if 'g_bar_star' in out: 
                        g_bar_star_next = process(out['g_bar_star'])
            
        ### 
        if save:
            if is_deq or ('only_next' != measure_default):
                np.save(f'{default_path}E_mae_prev.npy', E_mae_prev)
                np.save(f'{default_path}F_mae_prev.npy', F_mae_prev)  
                if save_atomwise_errors:
                    np.save(f'{default_path}F_mae_per_atom_prev.npy', F_mae_per_atom_prev)
                if save_preds:
                    np.save(f'{default_path}E_pred_prev.npy', E_pred_prev)
                    np.save(f'{default_path}F_pred_prev.npy', F_pred_prev)
                if is_deq and save_atomwise_residuals:
                    np.save(f'{default_path}forw_res_prev_per_atom.npy', forw_res_prev_per_atom)
                    np.save(f'{default_path}back_res_prev_per_atom.npy', back_res_prev_per_atom)
                if 'star' in save:
                    torch.save(z_star_prev, f'{default_path}z_star_prev.pt')
                    if g_star_prev is not None:
                        torch.save(g_star_prev, f'{default_path}g_star_prev.pt')
                    torch.save(g_bar_star_prev, f'{default_path}g_bar_star_prev.pt')
                
                np.save(f'{default_path}E_mae_curr.npy', E_mae_curr)
                np.save(f'{default_path}F_mae_curr.npy', F_mae_curr)  
                if save_atomwise_errors:
                    np.save(f'{default_path}F_mae_per_atom_curr.npy', F_mae_per_atom_curr)
                if save_preds:
                    np.save(f'{default_path}E_pred_curr.npy', E_pred_curr)
                    np.save(f'{default_path}F_pred_curr.npy', F_pred_curr)
                if is_deq and save_atomwise_residuals:
                    np.save(f'{default_path}forw_res_curr_per_atom.npy', forw_res_curr_per_atom)
                    np.save(f'{default_path}back_res_curr_per_atom.npy', back_res_curr_per_atom)
                if 'star' in save:
                    torch.save(z_star_curr, f'{default_path}z_star_curr.pt')
                    if g_star_curr is not None:
                        torch.save(g_star_curr, f'{default_path}g_star_curr.pt')
                    torch.save(g_bar_star_curr, f'{default_path}g_bar_star_curr.pt')
            
            if is_deq or measure_default:
                np.save(f'{default_path}E_mae_next.npy', E_mae_next)
                np.save(f'{default_path}F_mae_next.npy', F_mae_next)
                if save_atomwise_errors:
                    np.save(f'{default_path}F_mae_per_atom_next.npy', F_mae_per_atom_next)
                if save_preds:
                    np.save(f'{default_path}E_pred_next.npy', E_pred_next)
                    np.save(f'{default_path}F_pred_next.npy', F_pred_next)
                if is_deq and save_atomwise_residuals:
                    np.save(f'{default_path}forw_res_next_per_atom.npy', forw_res_next_per_atom)
                    np.save(f'{default_path}back_res_next_per_atom.npy', back_res_next_per_atom)
                if 'star' in save:
                    torch.save(z_star_next, f'{default_path}z_star_next.pt')
                    if g_star_next is not None:
                        torch.save(g_star_next, f'{default_path}g_star_next.pt')
                    torch.save(g_bar_star_next, f'{default_path}g_bar_star_next.pt')
        
        

    try:  # existence checks only; actual loading is lazy in the default/warm passes
        print('Trying to load default results...')
        assert not measure_default == True, 'user requested measure_defaults'

        if not save:
            raise FileNotFoundError('save=False: cannot rely on cached default artifacts')

        required = []

        # Explicit (non-DEQ) models: by default we only compute/save *_next.
        if is_deq or measure_default:
            required += [
                f'{default_path}F_mae_next.npy',
                f'{default_path}E_mae_next.npy',
            ]
            if save_atomwise_errors:
                required.append(f'{default_path}F_mae_per_atom_next.npy')
            if save_preds:
                required += [
                    f'{default_path}F_pred_next.npy',
                    f'{default_path}E_pred_next.npy',
                ]
            if is_deq and save_atomwise_residuals:
                required += [
                    f'{default_path}forw_res_next_per_atom.npy',
                    f'{default_path}back_res_next_per_atom.npy',
                ]

        if is_deq or ('only_next' != measure_default):
            required += [
                f'{default_path}F_mae_prev.npy',
                f'{default_path}E_mae_prev.npy',
                f'{default_path}F_mae_curr.npy',
                f'{default_path}E_mae_curr.npy',
            ]
            if save_atomwise_errors:
                required += [
                    f'{default_path}F_mae_per_atom_prev.npy',
                    f'{default_path}F_mae_per_atom_curr.npy',
                ]
            if save_preds:
                required += [
                    f'{default_path}F_pred_prev.npy',
                    f'{default_path}E_pred_prev.npy',
                    f'{default_path}F_pred_curr.npy',
                    f'{default_path}E_pred_curr.npy',
                ]
            if is_deq and save_atomwise_residuals:
                required += [
                    f'{default_path}forw_res_prev_per_atom.npy',
                    f'{default_path}back_res_prev_per_atom.npy',
                    f'{default_path}forw_res_curr_per_atom.npy',
                    f'{default_path}back_res_curr_per_atom.npy',
                ]

        if is_deq and save and ('star' in str(save)):
            required += [
                f'{default_path}z_star_prev.pt',
                f'{default_path}g_star_prev.pt',
                f'{default_path}g_bar_star_prev.pt',
                f'{default_path}z_star_curr.pt',
                f'{default_path}g_star_curr.pt',
                f'{default_path}g_bar_star_curr.pt',
                f'{default_path}z_star_next.pt',
                f'{default_path}g_star_next.pt',
                f'{default_path}g_bar_star_next.pt',
            ]

        if required and not _all_exist(required):
            missing = [p for p in required if not os.path.exists(p)]
            raise FileNotFoundError(f'Missing default artifacts ({len(missing)}/{len(required)}): {missing[0]}')

    except Exception as e: # else compute
        print('Received exception:', e)
        print('Computing default results...')
        
        status = run_default_pass()
        if status == 'surpassed n_batches available':
            return status

    if not is_deq or not measure_restricted or not return_warm:
        if not return_key:
            return
        out = _try_load_key(default_path, return_key, device=device)
        if out is not None:
            return out
        if return_key in locals():
            # e.g. save=False mode
            return locals()[return_key]
        raise FileNotFoundError(f'Could not load {return_key=} from {default_path}')

    if is_deq:
        for kind, degree in [('z', z), ('g', g), ('g_bar', g_bar)]:
            if degree >= 0:
                needed = [f'{default_path}{kind}_star_curr.pt']
                if degree == 1:
                    needed.append(f'{default_path}{kind}_star_prev.pt')
                missing = [p for p in needed if not os.path.exists(p)]
                assert not missing, (
                    f'{kind}={degree} warmstart requested but {kind}_star files not saved '
                    f'(model may not output {kind}_star): {missing}'
                )

    def run_warmstart_pass():        
        iterator = val_loader = loader_cached(partition=partition, limit=limit, **data_config)
        n_atoms = next(iter(val_loader))['_n_atoms'].detach()[0]
        n_batches = n_bs or len(val_loader)
        
        if only_batch is not None:
            if only_batch >= n_batches:
                print(f'Only batch {only_batch} requested, but only {n_batches} batches available. Exiting.')
                return 'surpassed n_batches available'
            else:
                n_batches = 1
                
            # extract the only batch that we want to process
            org = val_loader.dataset.subset_idx.copy()
            try:
                val_loader.dataset.subset_idx = org[only_batch*bsz:(only_batch+1)*bsz]
                iterator = [next(iter(val_loader))]
            finally:
                val_loader.dataset.subset_idx = org
        
        if True: ### stats and conversions
            nnpot, deq_obj = get_nnpot()
                
            n_conformations = min(len(val_loader.dataset), bsz * n_batches)
            z_shape = get_z_shape(nnpot, n_atoms, is_deq)
            n_atoms_int = int(n_atoms)

            process = lambda z: z.detach().clone().reshape(-1, *z_shape)
            mean_absolute_error = lambda a, b, bs=bsz: torch.mean(torch.abs(a.view(bs, -1)-b.view(bs, -1)), dim=1).detach().cpu().numpy()
            mean_absolute_error_per_atom = lambda a, b, bs=bsz: torch.mean(
                torch.abs(a.view(bs, n_atoms_int, -1) - b.view(bs, n_atoms_int, -1)),
                dim=2,
            ).detach().cpu().numpy()
            forces_to_numpy = lambda f, bs=bsz: f.view(bs, n_atoms_int, 3).detach().cpu().numpy()
            energy_to_numpy = lambda e, bs=bsz: e.view(bs).detach().cpu().numpy()
            res_to_numpy = lambda r, bs=bsz: r.view(bs, n_atoms_int).detach().cpu().numpy()
            degree_splits, basis_dim, has_degree_dim = get_degree_info(nnpot, is_deq)
            n_degrees = len(degree_splits)
            eps_res = 1e-12

            def residuals_per_degree(star, prev, bs=bsz):
                if basis_dim == 1:
                    diff = star.view(bs * n_atoms_int, -1) - prev.view(bs * n_atoms_int, -1)
                    abs_res = torch.linalg.norm(diff, dim=1).view(bs, n_atoms_int)
                    prev_norm = torch.linalg.norm(prev.view(bs * n_atoms_int, -1), dim=1).view(bs, n_atoms_int)
                    rel_res = abs_res / (prev_norm + eps_res)
                    return abs_res.detach().cpu().numpy(), rel_res.detach().cpu().numpy()
                star_v = star.view(bs * n_atoms_int, basis_dim, -1)
                prev_v = prev.view(bs * n_atoms_int, basis_dim, -1)
                diff = star_v - prev_v
                abs_parts = []
                rel_parts = []
                start = 0
                for count in degree_splits:
                    diff_g = diff[:, start:start + count, :]
                    prev_g = prev_v[:, start:start + count, :]
                    abs_g = torch.linalg.norm(diff_g.reshape(bs * n_atoms_int, -1), dim=1)
                    prev_norm = torch.linalg.norm(prev_g.reshape(bs * n_atoms_int, -1), dim=1)
                    abs_parts.append(abs_g)
                    rel_parts.append(abs_g / (prev_norm + eps_res))
                    start += count
                abs_res = torch.stack(abs_parts, dim=1).view(bs, n_atoms_int, n_degrees)
                rel_res = torch.stack(rel_parts, dim=1).view(bs, n_atoms_int, n_degrees)
                return abs_res.detach().cpu().numpy(), rel_res.detach().cpu().numpy()

            def residuals_overall(star, prev, bs=bsz):
                diff = star.view(bs, -1) - prev.view(bs, -1)
                abs_res = torch.linalg.norm(diff, dim=1)
                prev_norm = torch.linalg.norm(prev.view(bs, -1), dim=1)
                rel_res = abs_res / (prev_norm + eps_res)
                return abs_res.detach().cpu().numpy(), rel_res.detach().cpu().numpy()

            def diff_overall(a, b, bs=bsz):
                diff = a.view(bs, -1) - b.view(bs, -1)
                return torch.linalg.norm(diff, dim=1).detach().cpu().numpy()

            def diff_per_atom(a, b, bs=bsz):
                diff = a.view(bs, n_atoms_int, -1) - b.view(bs, n_atoms_int, -1)
                return torch.linalg.norm(diff, dim=2).detach().cpu().numpy()

            def diff_per_degree(a, b, bs=bsz):
                if basis_dim == 1:
                    return diff_per_atom(a, b, bs=bs)
                a_v = a.view(bs * n_atoms_int, basis_dim, -1)
                b_v = b.view(bs * n_atoms_int, basis_dim, -1)
                diff = a_v - b_v
                abs_parts = []
                start = 0
                for count in degree_splits:
                    diff_g = diff[:, start:start + count, :]
                    abs_g = torch.linalg.norm(diff_g.reshape(bs * n_atoms_int, -1), dim=1)
                    abs_parts.append(abs_g)
                    start += count
                return torch.stack(abs_parts, dim=1).view(bs, n_atoms_int, n_degrees).detach().cpu().numpy()

            def norm_overall(a, bs=bsz):
                return torch.linalg.norm(a.view(bs, -1), dim=1).detach().cpu().numpy()

            def norm_per_atom(a, bs=bsz):
                return torch.linalg.norm(a.view(bs, n_atoms_int, -1), dim=2).detach().cpu().numpy()

            def norm_per_degree(a, bs=bsz):
                if basis_dim == 1:
                    return norm_per_atom(a, bs=bs)
                a_v = a.view(bs * n_atoms_int, basis_dim, -1)
                norm_parts = []
                start = 0
                for count in degree_splits:
                    a_g = a_v[:, start:start + count, :]
                    norm_g = torch.linalg.norm(a_g.reshape(bs * n_atoms_int, -1), dim=1)
                    norm_parts.append(norm_g)
                    start += count
                return torch.stack(norm_parts, dim=1).view(bs, n_atoms_int, n_degrees).detach().cpu().numpy()
        
        star_next_available = False
        if is_deq and (save_atomwise_residuals or save_atomwise_degree_residuals):
            try:
                if 'z_star_next' not in locals():
                    z_star_next = _load_path_checked(f'{default_path}z_star_next.pt', device=device)
                if 'g_star_next' not in locals():
                    g_star_next = _load_path_checked(f'{default_path}g_star_next.pt', device=device)
                if 'g_bar_star_next' not in locals():
                    g_bar_star_next = _load_path_checked(f'{default_path}g_bar_star_next.pt', device=device)
                star_next_available = True
            except Exception as e:
                print('Could not load default *_star_next for warm diffs:', e)
                star_next_available = False

        print('Computing warm results...')
        if only_batch is None: ### inits
            E_mae_next_warm = np.full((n_conformations,), np.nan)
            F_mae_next_warm = np.full((n_conformations,), np.nan)
            if save_atomwise_errors:
                F_mae_per_atom_next_warm = np.full((n_conformations, n_atoms_int), np.nan)
            if save_preds:
                E_pred_next_warm = np.full((n_conformations,), np.nan)
                F_pred_next_warm = np.full((n_conformations, n_atoms_int, 3), np.nan)
            forw_res_next_warm = np.full((n_conformations,), np.nan)
            back_res_next_warm = np.full((n_conformations,), np.nan)
            if save_atomwise_residuals:
                forw_res_next_warm_per_atom = np.full((n_conformations, n_atoms_int), np.nan)
                back_res_next_warm_per_atom = np.full((n_conformations, n_atoms_int), np.nan)
                z_res_abs_next_warm = np.full((n_conformations,), np.nan)
                z_res_next_warm = np.full((n_conformations,), np.nan)
                if back_iter_on_g_bar:
                    g_bar_res_abs_next_warm = np.full((n_conformations,), np.nan)
                    g_bar_res_next_warm = np.full((n_conformations,), np.nan)
                else:
                    g_res_abs_next_warm = np.full((n_conformations,), np.nan)
                    g_res_next_warm = np.full((n_conformations,), np.nan)
                if star_next_available:
                    z_star_diff_next_warm = np.full((n_conformations,), np.nan)
                    z_star_diff_next_warm_per_atom = np.full((n_conformations, n_atoms_int), np.nan)
                    z_star_diff_rel_next_warm = np.full((n_conformations,), np.nan)
                    z_star_diff_rel_next_warm_per_atom = np.full((n_conformations, n_atoms_int), np.nan)
                    g_star_diff_next_warm = np.full((n_conformations,), np.nan)
                    g_star_diff_next_warm_per_atom = np.full((n_conformations, n_atoms_int), np.nan)
                    g_star_diff_rel_next_warm = np.full((n_conformations,), np.nan)
                    g_star_diff_rel_next_warm_per_atom = np.full((n_conformations, n_atoms_int), np.nan)
                    g_bar_star_diff_next_warm = np.full((n_conformations,), np.nan)
                    g_bar_star_diff_next_warm_per_atom = np.full((n_conformations, n_atoms_int), np.nan)
                    g_bar_star_diff_rel_next_warm = np.full((n_conformations,), np.nan)
                    g_bar_star_diff_rel_next_warm_per_atom = np.full((n_conformations, n_atoms_int), np.nan)
            if save_atomwise_degree_residuals:
                z_res_shape = (n_conformations, n_atoms_int, n_degrees) if has_degree_dim else (n_conformations, n_atoms_int)
                z_res_abs_next_warm_per_atom_deg = np.full(z_res_shape, np.nan)
                z_res_next_warm_per_atom_deg = np.full(z_res_shape, np.nan)
                if back_iter_on_g_bar:
                    g_bar_res_abs_next_warm_per_atom_deg = np.full(z_res_shape, np.nan)
                    g_bar_res_next_warm_per_atom_deg = np.full(z_res_shape, np.nan)
                else:
                    g_res_abs_next_warm_per_atom_deg = np.full(z_res_shape, np.nan)
                    g_res_next_warm_per_atom_deg = np.full(z_res_shape, np.nan)
                if star_next_available:
                    z_star_diff_next_warm_per_atom_deg = np.full(z_res_shape, np.nan)
                    g_star_diff_next_warm_per_atom_deg = np.full(z_res_shape, np.nan)
                    g_bar_star_diff_next_warm_per_atom_deg = np.full(z_res_shape, np.nan)
                    z_star_diff_rel_next_warm_per_atom_deg = np.full(z_res_shape, np.nan)
                    g_star_diff_rel_next_warm_per_atom_deg = np.full(z_res_shape, np.nan)
                    g_bar_star_diff_rel_next_warm_per_atom_deg = np.full(z_res_shape, np.nan)

            if save_star_restricted:
                z_star_next_warm = np.full((n_conformations, *z_shape), np.nan)
                g_star_next_warm = np.full((n_conformations, *z_shape), np.nan)
                    
        for j, batch in enumerate(tqdm(iterator)): ### run and save
            bs = len(batch['_n_atoms'])
            z_shape_batched = (bs*n_atoms, *z_shape[1:])
            R_shape_batched = (bs*n_atoms, 3)
            
            if j>=n_batches:
                break
            
            ### 'next' - test different warmstarting strategies and budgets
            inputs = _extract_successor_batch(batch, successor=2)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            
            F_next = inputs['forces'].clone()
            E_next = inputs['energy'].clone()
            
            # calculate true fixed point: what is the ideal we can approximate with warmstarts?
            R_next = inputs['_positions']

            ### second set of forward passes: fewer iterations, init with previous fix point
        
            ## warmstart z
            
            assert dh_dR_iter>=0 and (dh_dR_iter==0 or z==1)
            if z >= 0 and dh_dR_iter == 0:
                # constant/linear extrapolation of z_star from default pass (loaded lazily + cached
                # on CPU); move only this batch's slice to the GPU.
                inputs['z0'] = (
                    get_extrapolated_init(default_path, 'z', z)[j*bsz:j*bsz+bs]
                    .to(device)
                    .view(*z_shape_batched)
                )
            elif z == 1 and dh_dR_iter > 0:
                # Start from finite-difference in time and refine via JVPs wrt positions.
                # our initial guess is: z_curr - z_prev
                # We compute it as such: z_lin - z_curr == (2*curr - prev) - curr == curr - prev
                z_curr = (
                    get_extrapolated_init(default_path, 'z', 0)[j*bsz:j*bsz+bs]
                    .to(device)
                    .view(*z_shape_batched)
                )
                z_lin = get_extrapolated_init(default_path, 'z', 1)[j*bsz:j*bsz+bs].to(device).view(*z_shape_batched)
                dh_curr_t = dh_curr_0_fd = (z_lin - z_curr)

                # Lazily load current positions for this batch
                inputs_curr = _extract_successor_batch(batch, successor=1)
                inputs_curr = {key: value.to(device) for key, value in inputs_curr.items()}
                R_curr = inputs_curr['_positions']

                # obtain a linearized model around the current fixpoint and positions (for forward mode AD / JVPs)
                _, lin_f_curr = fw_linearize_spk(
                    nnpot.representation.f,
                    inputs_curr,
                    R_curr,
                    z_curr,
                )

                dR_curr = (R_next - R_curr).detach().clone()
                u_curr = lin_f_curr((dR_curr, torch.zeros_like(z_curr)))

                v_R = torch.zeros_like(dR_curr)
                for _ in range(dh_dR_iter):
                    dh_curr_t = u_curr + lin_f_curr((v_R, dh_curr_t))

                inputs['z0'] = z_curr + dh_curr_t
            else:
                assert z < 0 or dh_dR_iter == 0, 'Warmstart z with 1st degree, if you want to use dh_dR_iter.'
            
            ## warmstart g and g_bar
            
            assert g in [-1, 0, 1] and g_bar in [-1, 0, 1]
            if g >= 0:
                inputs['g0'] = (
                    get_extrapolated_init(default_path, 'g', g)[j*bsz:j*bsz+bs]
                    .to(device)
                    .view(*z_shape_batched)
                )
            if g_bar >= 0:
                inputs['g_bar0'] = (
                    get_extrapolated_init(default_path, 'g_bar', g_bar)[j*bsz:j*bsz+bs]
                    .to(device)
                    .view(*z_shape_batched)
                    .clone()
                )
            
            ## compute
            for key in ['z_star', 'g_star', 'g_bar_star']:
                if key in inputs: del inputs[key]
            deq_obj.forw_max_iter = forw_max_iter
            deq_obj.back_max_iter = back_max_iter
            deq_obj.return_zts = [forw_max_iter-1]
            deq_obj.return_gts = [back_max_iter-1]
            deq_obj.return_g_barts = [back_max_iter-1]
            
            if forw_max_iter==1 and 'z0' not in inputs:
                # with just one iteration, the deq class does not create the empty vector itself so we pass it:
                inputs['z0'] = torch.zeros(z_shape_batched, device=device)
                
            out = nnpot(inputs)
            next_forces_i = out['forces']
            next_energy_i = out['energy']
            
            # for res_save, res in [
            #     (forw_res_next_warm, out['forw_res']), 
            #     (back_res_next_warm, out['back_res'])]:
            #     res_save[j*bsz:j*bsz+bs] = gather_max_within_molecule(res[max(res)].cpu(), batch['_idx_m'])
            
            if only_batch is None:
                forw_res_next_warm[j*bsz:j*bsz+bs] = gather_max_within_molecule(out['forw_res'][max(out['forw_res'])].cpu(), batch['_idx_m'])
                back_res_next_warm[j*bsz:j*bsz+bs] = gather_max_within_molecule(out['back_res'][max(out['back_res'])].cpu(), batch['_idx_m'])
                if save_atomwise_residuals:
                    forw_res_next_warm_per_atom[j*bsz:j*bsz+bs] = res_to_numpy(out['forw_res'][max(out['forw_res'])], bs=bs)
                    back_res_next_warm_per_atom[j*bsz:j*bsz+bs] = res_to_numpy(out['back_res'][max(out['back_res'])], bs=bs)
                if save_atomwise_residuals or save_atomwise_degree_residuals:
                    z_prev = out['zts'][forw_max_iter - 1]
                    if back_iter_on_g_bar:
                        g_prev = out['g_barts'][back_max_iter - 1]
                    else:
                        g_prev = out['gts'][back_max_iter - 1]
                if save_atomwise_residuals:
                    z_abs_overall, z_rel_overall = residuals_overall(out['z_star'], z_prev, bs=bs)
                    z_res_abs_next_warm[j*bsz:j*bsz+bs] = z_abs_overall
                    z_res_next_warm[j*bsz:j*bsz+bs] = z_rel_overall
                    if back_iter_on_g_bar:
                        g_abs_overall, g_rel_overall = residuals_overall(out['g_bar_star'], g_prev, bs=bs)
                        g_bar_res_abs_next_warm[j*bsz:j*bsz+bs] = g_abs_overall
                        g_bar_res_next_warm[j*bsz:j*bsz+bs] = g_rel_overall
                    else:
                        g_abs_overall, g_rel_overall = residuals_overall(out['g_star'], g_prev, bs=bs)
                        g_res_abs_next_warm[j*bsz:j*bsz+bs] = g_abs_overall
                        g_res_next_warm[j*bsz:j*bsz+bs] = g_rel_overall
                    if star_next_available:
                        z_star_next_slice = z_star_next[j*bsz:j*bsz+bs]
                        z_star_diff_next_warm[j*bsz:j*bsz+bs] = diff_overall(out['z_star'], z_star_next_slice, bs=bs)
                        z_star_diff_next_warm_per_atom[j*bsz:j*bsz+bs] = diff_per_atom(out['z_star'], z_star_next_slice, bs=bs)
                        z_star_diff_rel_next_warm[j*bsz:j*bsz+bs] = z_star_diff_next_warm[j*bsz:j*bsz+bs] / (norm_overall(z_star_next_slice, bs=bs) + eps_res)
                        z_star_diff_rel_next_warm_per_atom[j*bsz:j*bsz+bs] = (
                            z_star_diff_next_warm_per_atom[j*bsz:j*bsz+bs] / (norm_per_atom(z_star_next_slice, bs=bs) + eps_res)
                        )
                        g_star_next_slice = g_star_next[j*bsz:j*bsz+bs]
                        g_star_diff_next_warm[j*bsz:j*bsz+bs] = diff_overall(out['g_star'], g_star_next_slice, bs=bs)
                        g_star_diff_next_warm_per_atom[j*bsz:j*bsz+bs] = diff_per_atom(out['g_star'], g_star_next_slice, bs=bs)
                        g_star_diff_rel_next_warm[j*bsz:j*bsz+bs] = g_star_diff_next_warm[j*bsz:j*bsz+bs] / (norm_overall(g_star_next_slice, bs=bs) + eps_res)
                        g_star_diff_rel_next_warm_per_atom[j*bsz:j*bsz+bs] = (
                            g_star_diff_next_warm_per_atom[j*bsz:j*bsz+bs] / (norm_per_atom(g_star_next_slice, bs=bs) + eps_res)
                        )
                        g_bar_star_next_slice = g_bar_star_next[j*bsz:j*bsz+bs]
                        g_bar_star_diff_next_warm[j*bsz:j*bsz+bs] = diff_overall(out['g_bar_star'], g_bar_star_next_slice, bs=bs)
                        g_bar_star_diff_next_warm_per_atom[j*bsz:j*bsz+bs] = diff_per_atom(out['g_bar_star'], g_bar_star_next_slice, bs=bs)
                        g_bar_star_diff_rel_next_warm[j*bsz:j*bsz+bs] = g_bar_star_diff_next_warm[j*bsz:j*bsz+bs] / (norm_overall(g_bar_star_next_slice, bs=bs) + eps_res)
                        g_bar_star_diff_rel_next_warm_per_atom[j*bsz:j*bsz+bs] = (
                            g_bar_star_diff_next_warm_per_atom[j*bsz:j*bsz+bs] / (norm_per_atom(g_bar_star_next_slice, bs=bs) + eps_res)
                        )
                if save_atomwise_degree_residuals:
                    z_abs, z_rel = residuals_per_degree(out['z_star'], z_prev, bs=bs)
                    z_res_abs_next_warm_per_atom_deg[j*bsz:j*bsz+bs] = z_abs
                    z_res_next_warm_per_atom_deg[j*bsz:j*bsz+bs] = z_rel
                    if back_iter_on_g_bar:
                        g_abs, g_rel = residuals_per_degree(out['g_bar_star'], g_prev, bs=bs)
                        g_bar_res_abs_next_warm_per_atom_deg[j*bsz:j*bsz+bs] = g_abs
                        g_bar_res_next_warm_per_atom_deg[j*bsz:j*bsz+bs] = g_rel
                    else:
                        g_abs, g_rel = residuals_per_degree(out['g_star'], g_prev, bs=bs)
                        g_res_abs_next_warm_per_atom_deg[j*bsz:j*bsz+bs] = g_abs
                        g_res_next_warm_per_atom_deg[j*bsz:j*bsz+bs] = g_rel
                    if star_next_available:
                        z_star_next_slice = z_star_next[j*bsz:j*bsz+bs]
                        z_star_diff_next_warm_per_atom_deg[j*bsz:j*bsz+bs] = diff_per_degree(out['z_star'], z_star_next_slice, bs=bs)
                        z_star_diff_rel_next_warm_per_atom_deg[j*bsz:j*bsz+bs] = (
                            z_star_diff_next_warm_per_atom_deg[j*bsz:j*bsz+bs] / (norm_per_degree(z_star_next_slice, bs=bs) + eps_res)
                        )
                        g_star_next_slice = g_star_next[j*bsz:j*bsz+bs]
                        g_star_diff_next_warm_per_atom_deg[j*bsz:j*bsz+bs] = diff_per_degree(out['g_star'], g_star_next_slice, bs=bs)
                        g_star_diff_rel_next_warm_per_atom_deg[j*bsz:j*bsz+bs] = (
                            g_star_diff_next_warm_per_atom_deg[j*bsz:j*bsz+bs] / (norm_per_degree(g_star_next_slice, bs=bs) + eps_res)
                        )
                        g_bar_star_next_slice = g_bar_star_next[j*bsz:j*bsz+bs]
                        g_bar_star_diff_next_warm_per_atom_deg[j*bsz:j*bsz+bs] = diff_per_degree(out['g_bar_star'], g_bar_star_next_slice, bs=bs)
                        g_bar_star_diff_rel_next_warm_per_atom_deg[j*bsz:j*bsz+bs] = (
                            g_bar_star_diff_next_warm_per_atom_deg[j*bsz:j*bsz+bs] / (norm_per_degree(g_bar_star_next_slice, bs=bs) + eps_res)
                        )
                
                if save_star_restricted:
                    z_star_next_warm[j*bsz:j*bsz+bs] = process(out['z_star']).cpu()
                    if 'g_star' in out: 
                        g_star_next_warm[j*bsz:j*bsz+bs] = process(out['g_star']).cpu()
                    
                F_mae_next_warm[j*bsz:j*bsz+bs] = mean_absolute_error(next_forces_i, F_next, bs=bs)
                E_mae_next_warm[j*bsz:j*bsz+bs] = mean_absolute_error(next_energy_i, E_next, bs=bs)
                if save_atomwise_errors:
                    F_mae_per_atom_next_warm[j*bsz:j*bsz+bs] = mean_absolute_error_per_atom(next_forces_i, F_next, bs=bs)
                if save_preds:
                    F_pred_next_warm[j*bsz:j*bsz+bs] = forces_to_numpy(next_forces_i, bs=bs)
                    E_pred_next_warm[j*bsz:j*bsz+bs] = energy_to_numpy(next_energy_i, bs=bs)
            else:
                forw_res_next_warm = gather_max_within_molecule(out['forw_res'][max(out['forw_res'])].cpu(), batch['_idx_m'])
                back_res_next_warm = gather_max_within_molecule(out['back_res'][max(out['back_res'])].cpu(), batch['_idx_m'])
                if save_atomwise_residuals:
                    forw_res_next_warm_per_atom = res_to_numpy(out['forw_res'][max(out['forw_res'])], bs=bs)
                    back_res_next_warm_per_atom = res_to_numpy(out['back_res'][max(out['back_res'])], bs=bs)
                if save_atomwise_residuals or save_atomwise_degree_residuals:
                    z_prev = out['zts'][forw_max_iter - 1]
                    if back_iter_on_g_bar:
                        g_prev = out['g_barts'][back_max_iter - 1]
                    else:
                        g_prev = out['gts'][back_max_iter - 1]
                if save_atomwise_residuals:
                    z_res_abs_next_warm, z_res_next_warm = residuals_overall(out['z_star'], z_prev, bs=bs)
                    if back_iter_on_g_bar:
                        g_bar_res_abs_next_warm, g_bar_res_next_warm = residuals_overall(out['g_bar_star'], g_prev, bs=bs)
                    else:
                        g_res_abs_next_warm, g_res_next_warm = residuals_overall(out['g_star'], g_prev, bs=bs)
                    if star_next_available:
                        z_star_next_slice = z_star_next
                        z_star_diff_next_warm = diff_overall(out['z_star'], z_star_next_slice, bs=bs)
                        z_star_diff_next_warm_per_atom = diff_per_atom(out['z_star'], z_star_next_slice, bs=bs)
                        z_star_diff_rel_next_warm = z_star_diff_next_warm / (norm_overall(z_star_next_slice, bs=bs) + eps_res)
                        z_star_diff_rel_next_warm_per_atom = z_star_diff_next_warm_per_atom / (norm_per_atom(z_star_next_slice, bs=bs) + eps_res)
                        g_star_next_slice = g_star_next
                        g_star_diff_next_warm = diff_overall(out['g_star'], g_star_next_slice, bs=bs)
                        g_star_diff_next_warm_per_atom = diff_per_atom(out['g_star'], g_star_next_slice, bs=bs)
                        g_star_diff_rel_next_warm = g_star_diff_next_warm / (norm_overall(g_star_next_slice, bs=bs) + eps_res)
                        g_star_diff_rel_next_warm_per_atom = g_star_diff_next_warm_per_atom / (norm_per_atom(g_star_next_slice, bs=bs) + eps_res)
                        g_bar_star_next_slice = g_bar_star_next
                        g_bar_star_diff_next_warm = diff_overall(out['g_bar_star'], g_bar_star_next_slice, bs=bs)
                        g_bar_star_diff_next_warm_per_atom = diff_per_atom(out['g_bar_star'], g_bar_star_next_slice, bs=bs)
                        g_bar_star_diff_rel_next_warm = g_bar_star_diff_next_warm / (norm_overall(g_bar_star_next_slice, bs=bs) + eps_res)
                        g_bar_star_diff_rel_next_warm_per_atom = g_bar_star_diff_next_warm_per_atom / (norm_per_atom(g_bar_star_next_slice, bs=bs) + eps_res)
                if save_atomwise_degree_residuals:
                    z_res_abs_next_warm_per_atom_deg, z_res_next_warm_per_atom_deg = residuals_per_degree(
                        out['z_star'], z_prev, bs=bs
                    )
                    if back_iter_on_g_bar:
                        g_bar_res_abs_next_warm_per_atom_deg, g_bar_res_next_warm_per_atom_deg = residuals_per_degree(
                            out['g_bar_star'], g_prev, bs=bs
                        )
                    else:
                        g_res_abs_next_warm_per_atom_deg, g_res_next_warm_per_atom_deg = residuals_per_degree(
                            out['g_star'], g_prev, bs=bs
                        )
                    if star_next_available:
                        z_star_next_slice = z_star_next
                        z_star_diff_next_warm_per_atom_deg = diff_per_degree(out['z_star'], z_star_next_slice, bs=bs)
                        z_star_diff_rel_next_warm_per_atom_deg = (
                            z_star_diff_next_warm_per_atom_deg / (norm_per_degree(z_star_next_slice, bs=bs) + eps_res)
                        )
                        g_star_next_slice = g_star_next
                        g_star_diff_next_warm_per_atom_deg = diff_per_degree(out['g_star'], g_star_next_slice, bs=bs)
                        g_star_diff_rel_next_warm_per_atom_deg = (
                            g_star_diff_next_warm_per_atom_deg / (norm_per_degree(g_star_next_slice, bs=bs) + eps_res)
                        )
                        g_bar_star_next_slice = g_bar_star_next
                        g_bar_star_diff_next_warm_per_atom_deg = diff_per_degree(out['g_bar_star'], g_bar_star_next_slice, bs=bs)
                        g_bar_star_diff_rel_next_warm_per_atom_deg = (
                            g_bar_star_diff_next_warm_per_atom_deg / (norm_per_degree(g_bar_star_next_slice, bs=bs) + eps_res)
                        )
                
                if save_star_restricted:
                    z_star_next_warm[j*bsz:j*bsz+bs] = process(out['z_star']).cpu()
                    if 'g_star' in out: 
                        g_star_next_warm[j*bsz:j*bsz+bs] = process(out['g_star']).cpu()
                    
                F_mae_next_warm = mean_absolute_error(next_forces_i, F_next, bs=bs)
                E_mae_next_warm = mean_absolute_error(next_energy_i, E_next, bs=bs)
                if save_atomwise_errors:
                    F_mae_per_atom_next_warm = mean_absolute_error_per_atom(next_forces_i, F_next, bs=bs)
                if save_preds:
                    F_pred_next_warm = forces_to_numpy(next_forces_i, bs=bs)
                    E_pred_next_warm = energy_to_numpy(next_energy_i, bs=bs)

        if save:
            np.save(f'{warm_path}E_mae_next.npy', E_mae_next_warm)
            np.save(f'{warm_path}F_mae_next.npy', F_mae_next_warm)
            if save_atomwise_errors:
                np.save(f'{warm_path}F_mae_per_atom_next.npy', F_mae_per_atom_next_warm)
            if save_preds:
                np.save(f'{warm_path}E_pred_next.npy', E_pred_next_warm)
                np.save(f'{warm_path}F_pred_next.npy', F_pred_next_warm)
            np.save(f'{warm_path}forw_res_next.npy', forw_res_next_warm)
            np.save(f'{warm_path}back_res_next.npy', back_res_next_warm)
            if save_atomwise_residuals:
                np.save(f'{warm_path}forw_res_next_per_atom.npy', forw_res_next_warm_per_atom)
                np.save(f'{warm_path}back_res_next_per_atom.npy', back_res_next_warm_per_atom)
                np.save(f'{warm_path}z_res_abs_next.npy', z_res_abs_next_warm)
                np.save(f'{warm_path}z_res_next.npy', z_res_next_warm)
                if back_iter_on_g_bar:
                    np.save(f'{warm_path}g_bar_res_abs_next.npy', g_bar_res_abs_next_warm)
                    np.save(f'{warm_path}g_bar_res_next.npy', g_bar_res_next_warm)
                else:
                    np.save(f'{warm_path}g_res_abs_next.npy', g_res_abs_next_warm)
                    np.save(f'{warm_path}g_res_next.npy', g_res_next_warm)
                if star_next_available:
                    np.save(f'{warm_path}z_star_diff_next_warm.npy', z_star_diff_next_warm)
                    np.save(f'{warm_path}z_star_diff_next_warm_per_atom.npy', z_star_diff_next_warm_per_atom)
                    np.save(f'{warm_path}z_star_diff_rel_next_warm.npy', z_star_diff_rel_next_warm)
                    np.save(f'{warm_path}z_star_diff_rel_next_warm_per_atom.npy', z_star_diff_rel_next_warm_per_atom)
                    np.save(f'{warm_path}g_bar_star_diff_next_warm.npy', g_bar_star_diff_next_warm)
                    np.save(f'{warm_path}g_bar_star_diff_next_warm_per_atom.npy', g_bar_star_diff_next_warm_per_atom)
                    np.save(f'{warm_path}g_bar_star_diff_rel_next_warm.npy', g_bar_star_diff_rel_next_warm)
                    np.save(f'{warm_path}g_bar_star_diff_rel_next_warm_per_atom.npy', g_bar_star_diff_rel_next_warm_per_atom)
                    np.save(f'{warm_path}g_star_diff_next_warm.npy', g_star_diff_next_warm)
                    np.save(f'{warm_path}g_star_diff_next_warm_per_atom.npy', g_star_diff_next_warm_per_atom)
                    np.save(f'{warm_path}g_star_diff_rel_next_warm.npy', g_star_diff_rel_next_warm)
                    np.save(f'{warm_path}g_star_diff_rel_next_warm_per_atom.npy', g_star_diff_rel_next_warm_per_atom)
            if save_atomwise_degree_residuals:
                np.save(f'{warm_path}z_res_abs_next_per_atom_deg.npy', z_res_abs_next_warm_per_atom_deg)
                np.save(f'{warm_path}z_res_next_per_atom_deg.npy', z_res_next_warm_per_atom_deg)
                if back_iter_on_g_bar:
                    np.save(f'{warm_path}g_bar_res_abs_next_per_atom_deg.npy', g_bar_res_abs_next_warm_per_atom_deg)
                    np.save(f'{warm_path}g_bar_res_next_per_atom_deg.npy', g_bar_res_next_warm_per_atom_deg)
                else:
                    np.save(f'{warm_path}g_res_abs_next_per_atom_deg.npy', g_res_abs_next_warm_per_atom_deg)
                    np.save(f'{warm_path}g_res_next_per_atom_deg.npy', g_res_next_warm_per_atom_deg)
                if star_next_available:
                    np.save(f'{warm_path}z_star_diff_next_warm_per_atom_deg.npy', z_star_diff_next_warm_per_atom_deg)
                    np.save(f'{warm_path}z_star_diff_rel_next_warm_per_atom_deg.npy', z_star_diff_rel_next_warm_per_atom_deg)
                    np.save(f'{warm_path}g_star_diff_next_warm_per_atom_deg.npy', g_star_diff_next_warm_per_atom_deg)
                    np.save(f'{warm_path}g_star_diff_rel_next_warm_per_atom_deg.npy', g_star_diff_rel_next_warm_per_atom_deg)
                    np.save(f'{warm_path}g_bar_star_diff_next_warm_per_atom_deg.npy', g_bar_star_diff_next_warm_per_atom_deg)
                    np.save(f'{warm_path}g_bar_star_diff_rel_next_warm_per_atom_deg.npy', g_bar_star_diff_rel_next_warm_per_atom_deg)
            if save_star_restricted:
                np.save(f'{warm_path}z_star_next.npy', z_star_next_warm)
                np.save(f'{warm_path}g_star_next.npy', g_star_next_warm)
        
        # computed warm artifacts (and saved if requested); fall through to unified return below
        pass


    try:
        print('Trying to load warm results...')
        assert measure_restricted is not True, 'user requested measure_restricted'

        if not save:
            raise FileNotFoundError('save=False: cannot rely on cached warm artifacts')

        required = [
            f'{warm_path}E_mae_next.npy',
            f'{warm_path}F_mae_next.npy',
            f'{warm_path}forw_res_next.npy',
            f'{warm_path}back_res_next.npy',
        ]
        if save_atomwise_errors:
            required.append(f'{warm_path}F_mae_per_atom_next.npy')
        if save_preds:
            required += [
                f'{warm_path}E_pred_next.npy',
                f'{warm_path}F_pred_next.npy',
            ]
        if save_atomwise_residuals:
            required += [
                f'{warm_path}forw_res_next_per_atom.npy',
                f'{warm_path}back_res_next_per_atom.npy',
                f'{warm_path}z_res_abs_next.npy',
                f'{warm_path}z_res_next.npy',
            ]
            if back_iter_on_g_bar:
                required += [
                    f'{warm_path}g_bar_res_abs_next.npy',
                    f'{warm_path}g_bar_res_next.npy',
                ]
            else:
                required += [
                    f'{warm_path}g_res_abs_next.npy',
                    f'{warm_path}g_res_next.npy',
                ]
        if save_atomwise_degree_residuals:
            required += [
                f'{warm_path}z_res_abs_next_per_atom_deg.npy',
                f'{warm_path}z_res_next_per_atom_deg.npy',
            ]
            if back_iter_on_g_bar:
                required += [
                    f'{warm_path}g_bar_res_abs_next_per_atom_deg.npy',
                    f'{warm_path}g_bar_res_next_per_atom_deg.npy',
                ]
            else:
                required += [
                    f'{warm_path}g_res_abs_next_per_atom_deg.npy',
                    f'{warm_path}g_res_next_per_atom_deg.npy',
                ]
        if save_star_restricted:
            required += [
                f'{warm_path}z_star_next.npy',
                f'{warm_path}g_star_next.npy',
            ]

        if required and not _all_exist(required):
            missing = [p for p in required if not os.path.exists(p)]
            raise FileNotFoundError(f'Missing warm artifacts ({len(missing)}/{len(required)}): {missing[0]}')

        # warm results exist, nothing to do
        pass
    except Exception as e: # else compute
        print('Received exception:', e)
        
        ## ... repetition from above.
        status = run_warmstart_pass()
        if status == 'surpassed n_batches available':
            return status

    if not return_key:
        return
    out = _try_load_key(warm_path, return_key, device=device)
    if out is not None:
        return out
    if return_key in locals():
        return locals()[return_key]
    raise FileNotFoundError(f'Could not load {return_key=} from {warm_path}')
