

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from ase import Atoms

### Functions in here are (redundant) tries 
### to align a molecule such that most atoms 
### move as little as possible for visualization. 
### (even if some explode away)

def center_and_align_traj(traj):
    aligned_traj = []

    for frame in traj:
        # Center by subtracting median per coordinate
        med = np.median(frame, axis=0)
        centered = frame - med

        # Compute distances to median-centered origin
        dists = np.linalg.norm(centered, axis=1)
        cutoff = np.percentile(dists, 90)
        core_atoms = centered[dists <= cutoff]

        # Compute covariance and eigen decomposition
        cov = core_atoms.T @ core_atoms
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        rotation = eigvecs[:, idx]

        # Rotate full frame
        rotated = centered @ rotation
        aligned_traj.append(rotated)

    return np.array(aligned_traj)

def kabsch_align(mobile, reference):
    # Subtract centroids
    ref_centered = reference - reference.mean(0)
    mob_centered = mobile - mobile.mean(0)

    # Covariance matrix and SVD
    C = mob_centered.T @ ref_centered
    V, _, Wt = np.linalg.svd(C)
    R = V @ Wt
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = V @ Wt

    # Rotate and translate
    aligned = (mob_centered @ R) + reference.mean(0)
    return aligned

def minimize_rmsd_traj(traj):
    traj_aligned = np.empty_like(traj)
    reference = traj[0]
    traj_aligned[0] = reference

    for t in range(1, len(traj)):
        traj_aligned[t] = kabsch_align(traj[t], reference)

    return traj_aligned

def kabsch_align_with_mask(mobile, reference, mask):
    # Apply mask
    ref_masked = reference[mask]
    mob_masked = mobile[mask]

    # Subtract centroids
    ref_centered = ref_masked - ref_masked.mean(0)
    mob_centered = mob_masked - mob_masked.mean(0)

    # Optimal rotation via SVD
    C = mob_centered.T @ ref_centered
    V, _, Wt = np.linalg.svd(C)
    R = V @ Wt
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = V @ Wt

    # Rotate all atoms using masked centering
    mob_all_centered = mobile - mob_masked.mean(0)
    aligned = (mob_all_centered @ R) + ref_masked.mean(0)
    return aligned

def compute_mask_90pct_close(traj_frame):
    # Geometric median approximation via coordinate-wise median
    median = np.median(traj_frame, axis=0)
    dists = np.linalg.norm(traj_frame - median, axis=1)
    cutoff = np.percentile(dists, 90)
    return dists <= cutoff

def minimize_rmsd_traj_auto_mask(traj):
    traj_aligned = np.empty_like(traj)
    reference = traj[0]
    traj_aligned[0] = reference

    mask = compute_mask_90pct_close(reference)

    for t in range(1, len(traj)):
        traj_aligned[t] = kabsch_align_with_mask(traj[t], reference, mask)

    return traj_aligned

def atoms_list_to_arrays(atoms_list):
    return np.array([atoms.get_positions() for atoms in atoms_list]), atoms_list[0].numbers

def arrays_to_atoms_list(positions_list, numbers):
    atoms_list = []
    for positions in positions_list:
        atoms = Atoms(positions=positions, numbers=numbers)
        atoms_list.append(atoms)
    return atoms_list