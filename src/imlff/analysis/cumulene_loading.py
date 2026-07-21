import os
import numpy as np
from schnetpack.transform import CastTo32, RemoveOffsets
import torch
from imlff.common.util import proj_path
from imlff.data.transforms import AllToAllNeighborList
from imlff.data.util import load_md17_data
from imlff.loading import load_model_from_wandb, download_ckpt_from_wandb
from tqdm import tqdm
from functools import lru_cache


@lru_cache(maxsize=20)
def cached_test_loader(molecule_i, batch_size=10, indices='all', dft=False):
    pre_setup_transforms = [AllToAllNeighborList(cutoff=None), CastTo32()]
    dm = load_md17_data(molecule=f'cumulene{molecule_i}{"_dft"*dft}_test', batch_size=batch_size, num_train=0, num_val=0, pre_setup_transforms=pre_setup_transforms)

    if indices != 'all':
        print(indices, list(indices))
        assert type(indices) in [list, tuple, np.ndarray]
        dm.test_idx = list(indices)
        dm._test_dataloader = None
        dm._test_dataset = dm.dataset.subset(dm.test_idx)
        dm._test_dataset.transforms = pre_setup_transforms
        dm._test_dataloader = None

    for i, t in enumerate(dm.test_transforms):
        if isinstance(t, RemoveOffsets):
            del dm.test_transforms[i]
            break
    print(dm.test_transforms)
    dl = dm.test_dataloader()
    print(dl.dataset, len(dl.dataset))
    return dl


def load_prep_model(model_tag, per_atom_output_key='energy_per_atom'):
    nnpot = load_model_from_wandb(model_tag)

    if type == 'deq':
        nnpot.representation.deq.implicit_diff = True
        nnpot.representation.deq.forw_tol = nnpot.representation.deq.back_tol = 0

    energy = nnpot.output_modules[0]
    assert energy.__class__.__name__ == 'Atomwise'
    energy.per_atom_output_key = per_atom_output_key
    if per_atom_output_key not in energy.model_outputs:
        energy.model_outputs.append(per_atom_output_key)

    nnpot.collect_outputs()
    nnpot.do_postprocessing = True
    return nnpot


def get_energy_offset(Rep='SO3Net', type='imp', model_i=(1,2,3,4,5), cutoff=5., model_tag=None, **kwargs):
    """Return the mean energy offset (kcal/mol per atom) subtracted during training.

    Reads the AddOffsets.mean buffer directly from the raw checkpoint to avoid GPU dependency.
    """
    if model_tag is None:
        from imlff.analysis.cumulene import get_model_tag
        model_tag = get_model_tag(Rep, type, model_i, cutoff, **kwargs)
    ckpt_path = download_ckpt_from_wandb(model_tag, verbose=False)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = state.get('state_dict', state)
    for key, val in sd.items():
        if 'postprocessors' in key and key.endswith('.mean'):
            return float(val.item())
    raise RuntimeError(f"No AddOffsets .mean found in checkpoint for model {model_tag}")


@lru_cache(maxsize=60)
def get_molecule_train_mean(molecule_i, dft=False, num_train=4500, num_val=450):
    """Per-atom energy mean of the cumulene train split, as RemoveOffsets used it at training.

    Multi-molecule training normalizes each molecule with its own train mean, but the
    checkpoint's AddOffsets stores only the count-weighted average across molecules
    (MultiMoleculeDataModule.get_stats) — so aligning predictions with raw labels
    requires this per-molecule mean.
    """
    pre_setup_transforms = [AllToAllNeighborList(cutoff=None), CastTo32()]
    dm = load_md17_data(molecule=f'cumulene{molecule_i}{"_dft"*dft}_train', batch_size=100,
                        num_train=num_train, num_val=num_val, pre_setup_transforms=pre_setup_transforms)
    mean, std = dm.get_stats('energy', True, False)
    return float(mean)


def get_n_a(batch):
    n_a = batch['_n_atoms']
    assert n_a.unique().numel() == 1
    n_a = n_a[0]
    return n_a


def calc_E_pred(nnpot, per_atom_output_key, test_loader, **config):
    Es = []
    Es_per_atom = []
    device = next(nnpot.parameters()).device

    for i, batch in enumerate(tqdm(test_loader)):
        del batch['energy']
        del batch['forces']

        for k in batch.keys():
            batch[k] = batch[k].to(device)
        batch.update(config)
        results = nnpot(batch)
        Es.append(results['energy'].detach().cpu().numpy())
        Es_per_atom.append(results[per_atom_output_key].view(-1, get_n_a(batch)).detach().cpu().numpy())

    E = np.concatenate(Es)
    E_per_atom = np.concatenate(Es_per_atom)
    return E, E_per_atom


def calc_EF_pred(nnpot, per_atom_output_key, test_loader, **config):
    """Like calc_E_pred but also returns predicted forces (concatenated over atoms)."""
    Es, Es_per_atom, Fs = [], [], []
    device = next(nnpot.parameters()).device

    for i, batch in enumerate(tqdm(test_loader)):
        del batch['energy']
        del batch['forces']

        for k in batch.keys():
            batch[k] = batch[k].to(device)
        batch.update(config)
        # forces come from autograd on positions, so no torch.no_grad here
        results = nnpot(batch)
        Es.append(results['energy'].detach().cpu().numpy())
        Es_per_atom.append(results[per_atom_output_key].view(-1, get_n_a(batch)).detach().cpu().numpy())
        Fs.append(results['forces'].detach().cpu().numpy())

    E = np.concatenate(Es)
    E_per_atom = np.concatenate(Es_per_atom)
    F = np.concatenate(Fs)  # (n_samples * n_atoms, 3)
    return E, E_per_atom, F


def read_E_label(test_loader):
    Es_label = []
    for i, batch in enumerate(tqdm(test_loader)):
        Es_label.append(batch['energy'].detach().cpu().numpy())
    E_label = np.concatenate(Es_label)
    return E_label


def read_EF_label(test_loader):
    Es_label, Fs_label = [], []
    for i, batch in enumerate(tqdm(test_loader)):
        Es_label.append(batch['energy'].detach().cpu().numpy())
        Fs_label.append(batch['forces'].detach().cpu().numpy())
    E_label = np.concatenate(Es_label)
    F_label = np.concatenate(Fs_label)  # (n_samples * n_atoms, 3)
    return E_label, F_label


def load_E_pred(model_path):
    E = np.load(model_path + 'E.npy')
    E_per_atom = np.load(model_path + 'E_per_atom.npy')
    return E, E_per_atom

def save_E_pred(model_path, E, E_per_atom):
    os.makedirs(model_path, exist_ok=True)
    np.save(model_path + 'E.npy', E)
    np.save(model_path + 'E_per_atom.npy', E_per_atom)

def load_E_label(model_path):
    return np.load(model_path + 'E.npy')

def save_E_label(model_path, E_label):
    os.makedirs(model_path, exist_ok=True)
    np.save(model_path + 'E.npy', E_label)


@lru_cache(maxsize=100)
def load_or_calc(Rep='SO3Net', type='imp', model_i=(1,2,3,4,5), cutoff=5., molecule_i=None, latest=False, norm='auto',
                 recalc='auto', model_tag=None, train_forces=0.95, jac_reg=0., train_dft=False, verbose=False, **kwargs_scf):
    assert molecule_i is not None, 'molecule_i must be specified'

    if model_tag is None:
        from imlff.analysis.cumulene import get_model_tag
        model_tag = get_model_tag(Rep, type, model_i, cutoff, latest=latest, norm=norm, train_forces=train_forces, jac_reg=jac_reg, train_dft=train_dft, verbose=verbose)

    molecule_path = proj_path(f'results/cumulene_{molecule_i}/', verbose=False)
    model_path = molecule_path + model_tag.replace("/", "_") + '/'
    try:
        assert recalc is not True
        if 'latest' in model_tag:
            assert recalc != 'auto'
        E, E_per_atom = load_E_pred(model_path)
    except:
        print('Calculating E_pred...')
        per_atom_output_key = 'energy_per_atom'
        nnpot = load_prep_model(model_tag, per_atom_output_key)
        test_loader = cached_test_loader(molecule_i, dft=train_dft)
        E, E_per_atom = calc_E_pred(nnpot, per_atom_output_key, test_loader)
        save_E_pred(model_path, E, E_per_atom)
    return E, E_per_atom


def load_or_read_out_label(molecule_i, dft=False):
    molecule_path = proj_path(f'results/cumulene_{molecule_i}{"_dft"*dft}/', verbose=False)
    try:
        E_label = load_E_label(molecule_path)
    except FileNotFoundError:
        print('Calculating E_label...')
        test_loader = cached_test_loader(molecule_i, dft=dft)
        E_label = read_E_label(test_loader)
        save_E_label(molecule_path, E_label)
    return E_label


def load_or_calc_mae(Rep='SO3Net', type='imp', model_i=(1,2,3,4,5), cutoff=5., molecule_i=None,
                     latest=False, norm='auto', recalc='auto', model_tag=None, train_forces=0.95,
                     jac_reg=0., train_dft=False, add_offsets=True, results_dir='results',
                     use_tmp=False, verbose=False, **kwargs_scf):
    """Per-molecule energy & force MAE for one cumulene model.

    Energy is aligned to the raw labels exactly as plot_E_per_molecule does (swap the
    global AddOffsets mean for the per-molecule train mean); forces are offset-invariant.
    MAEs are in the dataset's units: eV for energy, eV/Ang for forces (the cumulene
    DB stores eV and the loader applies no unit conversion).

    Results are cached as run_test.py does: results/cumulene_{molecule_i}/{model_str}/test.npz
    with keys test_energy_MAE / test_forces_MAE.
    """
    assert molecule_i is not None, 'molecule_i must be specified'

    if model_tag is None:
        from imlff.analysis.cumulene import get_model_tag
        model_tag = get_model_tag(Rep, type, model_i, cutoff, latest=latest, norm=norm,
                                  train_forces=train_forces, jac_reg=jac_reg, train_dft=train_dft,
                                  verbose=verbose)

    model_str = model_tag.replace("/", "_")
    results_path = proj_path(f'{results_dir}/cumulene_{molecule_i}/{model_str}/test.npz',
                             use_tmp=use_tmp, verbose=False)

    if recalc is not True and os.path.exists(results_path):
        npz = np.load(results_path)
        return float(npz['test_energy_MAE']), float(npz['test_forces_MAE'])

    print(f'Calculating MAEs for molecule_i={molecule_i}, {model_tag}...')
    from imlff.analysis.cumulene import get_Z
    per_atom_output_key = 'energy_per_atom'
    nnpot = load_prep_model(model_tag, per_atom_output_key)
    test_loader = cached_test_loader(molecule_i, dft=train_dft)
    E_pred, _, F_pred = calc_EF_pred(nnpot, per_atom_output_key, test_loader)
    E_label, F_label = read_EF_label(test_loader)

    if add_offsets:
        # training removed this molecule's own mean; AddOffsets restored only the
        # global average — swap the global mean for the per-molecule one
        offset = get_energy_offset(model_tag=model_tag)
        mean_i = get_molecule_train_mean(molecule_i, dft=train_dft)
        E_pred = E_pred - (offset - mean_i) * len(get_Z(molecule_i))

    energy_mae = float(np.mean(np.abs(E_pred - E_label)))
    forces_mae = float(np.mean(np.abs(F_pred - F_label)))

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    np.savez(results_path, test_energy_MAE=energy_mae, test_forces_MAE=forces_mae)
    return energy_mae, forces_mae
