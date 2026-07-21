import shutil
import torch
import os
import schnetpack.transform as trn

from multiprocessing import Pool
from functools import partial
import pytorch_lightning as pl
from schnetpack.datasets import MD17, rMD17, MD22
from imlff.common.util import proj_path
from imlff.data.cumulene import Cumulene
from imlff.data.info import molecule_to_dataset
from imlff.data.transforms import AllToAllNeighborList
from schnetpack.transform.neighborlist import TorchNeighborList
from imlff.scripts.generate_splits import get_split_file_descriptor
      
from functools import lru_cache

@lru_cache(maxsize=32)
def load_data_cached(**kwargs):
    return load_md17_data(**kwargs)

def split_file_path_for(
    molecule,
    num_train,
    num_val,
    split_id,
    *,
    data_dir='data',
    Dataset=None,
    use_tmp='auto',
    verbose=True,
):
    """Resolve the split-file path for a (molecule, sizes, split_id) configuration.

    This mirrors the path logic inside ``load_one_datamodule`` so callers that only
    need the train/val/test index split (saved as an ``.npz`` with keys
    ``train_idx``/``val_idx``/``test_idx``) can find it without constructing a
    datamodule. Keep this in sync with ``load_one_datamodule`` below.
    """
    if Dataset is None:
        Dataset = molecule_to_dataset(molecule, name=False)
    dataset_str = Dataset.__name__


    splits_dir = 'splits'
    
    return proj_path(
        data_dir,
        dataset_str,
        splits_dir,
        get_split_file_descriptor(molecule, num_train, num_val, split_id),
        use_tmp=use_tmp,
        verbose=verbose,
    )


def load_md17_data(molecule, **kwargs):
    if isinstance(molecule,  (list, tuple)):
        mm = MultiMoleculeDataModule(molecule=molecule, **kwargs)
        mm.prepare_data()
        mm.setup()
        return mm
    elif type(molecule) is str:
        return load_one_datamodule(molecule=molecule, **kwargs)
    else:
        raise ValueError(f"Unsupported type for molecule: {type(molecule)}. Expected str or list of str.")

def load_one_datamodule(data_dir='data', Dataset=None, molecule='ethanol', num_train=950, num_val=1000, batch_size=100,
                   split_id=0, split_file_path=None, load_successors=False, # (load_successors only active in warmstart analysis)
                   num_workers=None, use_tmp='auto', duplicate_for_run=False, adj_cutoff=1e8, prepare=True, pin_memory=None,
                   property_units=None, distance_unit=None, **dataset_kwargs):
    assert Dataset in [MD17, rMD17, MD22, Cumulene, None], "Only MD17, rMD17, MD22, Cumulene are supported."
    if Dataset is None: Dataset = molecule_to_dataset(molecule, name=False)
    dataset_str = Dataset.__name__
    
    if split_file_path is None:
        split_file_path = split_file_path_for(
            molecule, num_train, num_val, split_id,
            data_dir=data_dir, Dataset=Dataset,
        )

        if True:
            assert os.path.exists(split_file_path), f"Split file not found: {split_file_path}\n"\
                "You can create new split files by temporarily ommiting this assertion, but please push them to GH such that we are aligned."
                
    print(f"Using split file: {split_file_path}")
    
    if adj_cutoff>1e5:
        nl = AllToAllNeighborList(cutoff=None)
        print('\n'*0 + f"Using AllToAllNeighborList.")
    else:
        nl = TorchNeighborList(cutoff=adj_cutoff)
        print('\n'*0 + f"Using TorchNeighborList with cutoff {adj_cutoff}.")

    transforms=[
            # cutoff doesn't matter here as it won't be used in the transform
            nl,
            trn.RemoveOffsets(Dataset.energy, remove_mean=True, remove_atomrefs=False),
            trn.CastTo32()
        ]
    
    if Dataset is rMD17:
        Dataset = partial(Dataset, split_id=split_id)
        if split_id is not None:
            assert 0<num_train<1000, "num_train must be positive"
            assert 0<num_val<1000, "num_val must be positive"
            assert num_train+num_val<=1000, "num_train+num_val must be less than 1000 for rMD17 predefined splits."

    data_path = proj_path(data_dir, dataset_str, use_tmp=use_tmp, verbose=False) + f'/{molecule}.db'
    print('DATA PATH:\n', data_path)
    
    if duplicate_for_run:
        assert os.path.exists(data_path), f"Download the dataset once without duplicate_for_run first."
        
        postfix = os.urandom(4).hex()
        tmp_path = f'/tmp/imlff-tmp-{molecule}_{postfix}.db'
        print('TMP PATH:\n', tmp_path)
        
        shutil.copy(data_path, tmp_path)
        data_path = tmp_path
            
    if property_units is None:
        property_units = {"energy": "kcal/mol", "forces": "kcal/mol/Ang"}
    if distance_unit is None:
        distance_unit = "Ang"

    num_workers = num_workers if num_workers is not None else (16 if torch.cuda.is_available() else 0)
    print('Initializing with batch_size:::', batch_size)
    data = Dataset(
        data_path,
        molecule=molecule,
        batch_size=batch_size,
        val_batch_size=batch_size,
        num_train=num_train,
        num_val=num_val,
        transforms=transforms,
        num_workers=num_workers,
        pin_memory=pin_memory if (pin_memory is not None) else torch.cuda.is_available(), # set to false, when not using a GPU
        split_file=split_file_path,
        load_successors=load_successors,
        property_units=property_units,
        distance_unit=distance_unit,
        **dataset_kwargs
    )
    if prepare:
        data.prepare_data()
        data.setup()
    
    # if we loaded the data for the first time, and into temp, pull it over into home directory
    assert os.path.exists(data_path), f"Something went wrong in dataset creation"
    data_path_non_temp = proj_path(data_dir, dataset_str, use_tmp=False,  verbose=False) + f'/{molecule}.db'
    if not os.path.exists(data_path_non_temp):
        shutil.copy(data_path, data_path_non_temp)
        
    print('NON-TMP PATH:\n', data_path_non_temp)
    return data

## Only for development purposes

class MultiDatasetLoader:
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        self.iters = [iter(dl) for dl in dataloaders]

    def __iter__(self):
        while True:
            batch_parts = []
            for i, it in enumerate(self.iters):
                try:
                    part = next(it)
                except StopIteration:
                    self.iters[i] = iter(self.dataloaders[i])
                    part = next(self.iters[i])
                batch_parts.append(part)
                
            merged = {}
            atoms_so_far = 0
            mols_so_far = 0
            for part in batch_parts:
                for k, v in part.items():
                    if k == '_idx_m':
                        v += mols_so_far
                    if k in ['_idx_i', '_idx_j']:
                        v += atoms_so_far
                    merged.setdefault(k, []).append(v)
                    
                n_a = part['_n_atoms']
                atoms_so_far += n_a.sum().item()
                mols_so_far  += len(n_a)
            yield {k: torch.cat(v, dim=0) for k, v in merged.items()}

    def __len__(self):
        return min(len(dl) for dl in self.dataloaders)
    
class MultiMoleculeDataModule(pl.LightningDataModule):
    def __init__(self, molecule, num_train, num_val, **kwargs):
        super().__init__()
        self.molecules  = molecule  if isinstance(molecule,  (list, tuple)) else [molecule]
        self.num_trains = num_train if isinstance(num_train, (list, tuple)) else [num_train] * len(self.molecules)
        self.num_vals   = num_val   if isinstance(num_val,   (list, tuple)) else [num_val] * len(self.molecules)
        self.kwargs = kwargs
        self.datasets = []

    def prepare_data(self):
        def _load(args):
            mol, ntr, nval = args
            return load_one_datamodule(molecule=mol, num_train=ntr, num_val=nval, **self.kwargs)

        self.datasets = [_load(args) for args in zip(self.molecules, self.num_trains, self.num_vals)]
        # with Pool() as pool:
        #     self.datasets = pool.map(_load, zip(self.molecules, self.num_trains, self.num_vals))

    def setup(self, stage=None):
        self.train_loaders = [dl for ds in self.datasets if (dl:=ds.train_dataloader()) is not None]
        self.val_loaders   = [dl for ds in self.datasets if (dl:=ds.val_dataloader())   is not None]
        self.test_loaders  = [dl for ds in self.datasets if (dl:=ds.test_dataloader())  is not None]

    def train_dataloader(self):
        return MultiDatasetLoader(self.train_loaders) if self.train_loaders else None

    def val_dataloader(self):
        return MultiDatasetLoader(self.val_loaders) if self.val_loaders else None

    def test_dataloader(self):
        return MultiDatasetLoader(self.test_loaders) if self.test_loaders else None
    
    def get_stats(self, property_name, is_extensive, add_atomrefs):
        means, stds, counts = [], [], []        
        
        for ds in self.datasets:
            mean, std = ds.get_stats(property_name, is_extensive, add_atomrefs)
            means.append(mean)
            stds.append(std)
            counts.append(len(ds.train_idx))

        # Compute overall mean
        count_total = sum(counts)
        mean_total = sum(m * n for m, n in zip(means, counts)) / count_total

        # Compute overall variance using the law of total variance
        # var_total = (
        #     sum(n * (v + (m - mean_total)**2) for m, v, n in zip(means, vars, counts)) / count_total
        # )
        std_total = (
            sum(n * (s**2 + (m - mean_total)**2) for m, s, n in zip(means, stds, counts)) / count_total
        ).sqrt() 
        
        return mean_total, std_total
