import torch
from schnetpack.transform.neighborlist import NeighborListTransform


class AllToAllNeighborList(NeighborListTransform):
    """
    Calculate a full neighbor list for all atoms in the system.
    """

    def _build_neighbor_list(self, Z, positions, cell, pbc, cutoff):
        n_atoms = Z.shape[0]
        idx_i = torch.arange(n_atoms).repeat_interleave(n_atoms)
        idx_j = torch.arange(n_atoms).repeat(n_atoms)

        mask = idx_i != idx_j
        idx_i = idx_i[mask]
        idx_j = idx_j[mask]

        offset = torch.zeros(n_atoms * (n_atoms - 1), 3, dtype=positions.dtype, device=positions.device)
        return idx_i, idx_j, offset
