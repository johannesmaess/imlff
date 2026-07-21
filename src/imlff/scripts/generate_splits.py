import os
import numpy as np
from schnetpack.data.splitting import random_split
import torch
from imlff.common.util import proj_path
from imlff.data.info import molecule_to_dataset, molecule_to_n_conformations, cumulene_conformations_per_molecule, md17_conformations_per_molecule, md22_conformations_per_molecule

def get_split_file_descriptor(molecule, num_train, num_val, split_id):
    return f'{molecule}_t{num_train}_v{num_val}_split{split_id}.npz'

def check_no_overlap(arr1, arr2, arr3):
    return (
        len(np.intersect1d(arr1, arr2)) == 0 and
        len(np.intersect1d(arr1, arr3)) == 0 and
        len(np.intersect1d(arr2, arr3)) == 0
    )

def generate_dataset_splits(data_dir, molecule,
                            n_splits=5,
                            train_set_sizes=[100, 950, 1000, 5000, 10000, 50000],
                            val_set_sizes=[1000, 100, 50, 10, 0],
                            min_test_set_size=48999,
                            min_val_set_size=1000,
                            min_train_set_size=10000,
                            overwrite=False,
                            ):
    
    """
    read_from_larger_split_file: Optional[tuple[int, int]]
    """
    splits_dir = 'splits'
    dataset_dir = molecule_to_dataset(molecule, name=True)
    
    dsize = molecule_to_n_conformations(molecule)
    max_num_train = max(max(train_set_sizes), min_train_set_size)
    max_num_val = max(max(val_set_sizes), min_val_set_size)
  
    assert max_num_train + max_num_val + min_test_set_size <= dsize, f"Sum of train, val, and min test set sizes exceeds total number of conformations for {molecule}: {max_num_train=} + {max_num_val=} + {min_test_set_size=} = {max_num_train+max_num_val+min_test_set_size} <= {dsize=}"

    reduced_dsize = dsize
    
    split_dir_path = os.path.join(data_dir, dataset_dir, splits_dir)
    if not os.path.exists(split_dir_path):
        os.makedirs(split_dir_path)
    
    for split_id in range(n_splits):
        torch.manual_seed(split_id)
        train_split_full, val_split_full, *_ = random_split(reduced_dsize, max_num_train, max_num_val, -1)
        train_split_full, val_split_full = np.array(train_split_full), np.array(val_split_full)
                
        test_split = np.setdiff1d(np.arange(dsize), np.concatenate([train_split_full, val_split_full]))
        
        for num_train in train_set_sizes:
            for num_val in val_set_sizes:
                split_file_path = os.path.join(split_dir_path, get_split_file_descriptor(molecule, num_train, num_val, split_id))
                if not overwrite:
                    assert not os.path.exists(split_file_path), f"Split file already exists: {split_file_path}. Overwrite is set to False. (Don't change splits because you might have to rerun experiments.)"
                    
                train_split = train_split_full[:num_train]
                val_split = val_split_full[:num_val]
                
                np.savez(split_file_path, train_idx=train_split, val_idx=val_split, test_idx=test_split)

if __name__ == '__main__':
    data_dir = proj_path('data', use_tmp=False)
    print(data_dir)
        
    for molecule in md17_conformations_per_molecule.keys():
        
        generate_dataset_splits(data_dir, molecule, overwrite=True, 
                                # train_set_sizes=[950, 1000, 5000, 10000], min_test_set_size=63109, min_val_set_size=1000, min_train_set_size=10000,
                                train_set_sizes=[10, 950, 1000, 5000, 10000], val_set_sizes=[10, 50, 1000], min_test_set_size=10000, min_val_set_size=1000, min_train_set_size=10000,
                                n_splits=1,
                                )
    
    # for molecule, min_train_set_size, min_val_set_size, min_test_set_size in [

    #     ('Ac-Ala3-NHMe', 6000, 1000, 69109),
    #     ('DHA', 8000, 1000, 60753),
    #     ('AT-AT', 5000, 1000, 14001),
    #     ('stachyose', 8000, 1000, 18272),
    #     ('AT-AT-CG-CG', 5000, 1000, 4153),
    #     ('buckyball-catcher', 3000, 1000, 2102),
    #     ('double-walled_nanotube', 3000, 1000, 1032),
    #     ]:
    #     generate_dataset_splits(data_dir, molecule, overwrite=True,
    #                             # train_set_sizes=[100, 950, 1000, 5000, 10000], min_test_set_size=58753, min_val_set_size=1000, min_train_set_size=10000, # run for DHA
    #                             # train_set_sizes=[100, 950, 1000, 5000], min_test_set_size=21272, min_val_set_size=1000, min_train_set_size=5000,
    #                             train_set_sizes=[i for i in [10, 100, 200, 450, 600, 800, 950, 2000, 1000, 3000, 5000, 8000, 10000] if i<=min_train_set_size],
    #                             val_set_sizes=[i for i in [0, 10, 50, 100, 500, 1000] if i <= min_val_set_size], 
    #                             min_train_set_size=min_train_set_size, # run for 'Ac-Ala3-NHMe', 'DHA' with extra train=10000
    #                             min_val_set_size=min_val_set_size, 
    #                             min_test_set_size=min_test_set_size,
    #                             n_splits=1,
    #                             )
    
    # # For Cumulene train (incl val) and test are already split
    # for i in range(1,9):
    #     molecule = f'cumulene{i}_train'
    #     generate_dataset_splits(data_dir, molecule, overwrite=True,
    #                             train_set_sizes=[10, 4500], val_set_sizes=[10, 450], min_test_set_size=0, min_val_set_size=450, min_train_set_size=4500, # tiny
    #                             n_splits=1,
    #                             )
    # for i in range(1,9):
    #     molecule = f'cumulene{i}_test'
    #     generate_dataset_splits(data_dir, molecule, overwrite=True,
    #                             train_set_sizes=[0], val_set_sizes=[0], min_test_set_size=1000, min_val_set_size=0, min_train_set_size=0, # tiny
    #                             n_splits=1,
    #                             )
    
    
    
    
_resulting_test_split_sizes = {
    'AT-AT-CG-CG': 4153,
    'Ac-Ala3-NHMe': 74109,
    'DHA': 58753,
    'buckyball-catcher': 2102,
    'double-walled_nanotube': 1032,
    'stachyose': 16272,
    'aspirin': 160762,
    'azobenzene': 48999,
    'benzene': 576983,
    'ethanol': 504092,
    'malonaldehyde': 942237,
    'naphthalene': 275250,
    'paracetamol': 55490,
    'salicylic_acid': 269231,
    'toluene': 391790,
    'uracil': 82770
}