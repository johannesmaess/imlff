import numpy as np
import torch
import os
from typing import Optional, List
from filelock import FileLock
from schnetpack.data import AtomsDataModule, ASEAtomsData
from ase import Atoms

class Cumulene(AtomsDataModule):

    energy = "energy"
    forces = "forces"
    
    def __init__(
        self,
        datapath: str,
        molecule: str,
        split_file: str,
        batch_size: int,
        num_train: Optional[int] = None,
        num_val: Optional[int] = None,
        num_workers: int = 8,
        transforms: Optional[List[torch.nn.Module]] = None,
        pre_setup_transforms: Optional[List[torch.nn.Module]] = None,
        only_converged: bool = True,
        **kwargs
    ):
        super().__init__(
            datapath=datapath,
            batch_size=batch_size,
            num_train=num_train,
            num_val=num_val,
            split_file=split_file,
            num_workers=num_workers,
            transforms=transforms,
            **kwargs
        )
        self.split_file = split_file
        self.molecule = molecule
        assert self.datapath.endswith('.db')
        # replace:
        self.npzpath = self.datapath.replace('.db', '.npz')
        
        self.only_converged = only_converged
        self.pre_setup_transforms = pre_setup_transforms

    def setup(self, stage: Optional[str] = None):
        # check if the database has been created
        if not os.path.exists(self.datapath):
            # Load the NPZ data
            data = np.load(self.npzpath)
            
            # Create ASE Atoms objects
            atoms_list = [Atoms(positions=R, numbers=data['z']) for R in data['R']]
            
            # Prepare properties
            properties_list = [{'energy': E, 'forces': F} for E, F in zip(data['E'], data['F'])]
            
            # Filter out non-converged calculations if specified
            if 'converged' not in data:
                self.only_converged = False
            if self.only_converged:
                # Filter out atoms that are not converged
                atoms_list =      [atoms for atoms, converged in zip(atoms_list,      data['converged']) if converged]
                properties_list = [props for props, converged in zip(properties_list, data['converged']) if converged]
            
            # Create ASEAtomsData database
            self.dataset = ASEAtomsData.create(
                self.datapath,
                distance_unit="Ang",
                property_unit_dict={
                    "energy": "eV",
                    "forces": "eV/Ang",
                }
            )
            
            # Add systems to the database
            self.dataset.add_systems(properties_list, atoms_list)
        else:
            self.dataset = ASEAtomsData(self.datapath)
            
        try:
            # Load split indices
            split_data = np.load(self.split_file)
            self.train_idx = split_data['train_idx'].tolist()
            self.val_idx = split_data['val_idx'].tolist()
            if 'test_idx' in split_data:
                self.test_idx = split_data['test_idx'].tolist()
            else:
                self.test_idx = []
        
        except Exception as e:
            print('Could not read split file, setting all indices to [0]. Error:', e)
            self.train_idx = [0]
            self.val_idx = [0]
            self.test_idx = [0]
            

        # Partition dataset
        self._train_dataset = self.dataset.subset(self.train_idx)
        self._val_dataset = self.dataset.subset(self.val_idx)
        if self.test_idx:
            self._test_dataset = self.dataset.subset(self.test_idx)
        else:
            self._test_dataset = None

        if self.pre_setup_transforms is not None:
            self._train_dataset.transforms = self.pre_setup_transforms
            self._val_dataset.transforms = self.pre_setup_transforms
            self._test_dataset.transforms = self.pre_setup_transforms
        else:
            self._setup_transforms()
            
