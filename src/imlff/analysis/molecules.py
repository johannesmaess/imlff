import torch
from schnetpack.transform.neighborlist import TorchNeighborList
from imlff.data.transforms import AllToAllNeighborList


def create_batch(pos, Z=None, connectivity=None, device='cpu', idx_m=None, **kwargs):
    if type(pos) is int:
        pos = torch.randn((pos, 3), device=device)
    
    n_atoms_total = pos.shape[0]
    if Z is None:
        Z = torch.full((n_atoms_total,), 6, device=device)
        # Z = torch.full((pos.shape[0],), 1)
        
    if idx_m is None:
        idx_m = torch.zeros((n_atoms_total,), dtype=int, device=device)

    batch = {
    '_atomic_numbers': Z.to(device),
    '_n_atoms': len(Z),
    '_positions': pos.to(device),
    '_cell': torch.zeros((3,3), device=device),
    '_pbc': torch.tensor([False, False, False], device=device),
    '_idx_m': idx_m.to(device),
    **kwargs
    }
    if connectivity is None:
        batch_with_neighbourlist =  AllToAllNeighborList(None)(batch)
        batch_with_neighbourlist = {k:v.to(device) if type(v) is torch.Tensor else v for k,v in batch_with_neighbourlist.items()}
        return batch_with_neighbourlist
    if type(connectivity) is float:
        batch_with_neighbourlist = TorchNeighborList(cutoff=connectivity)(batch)
        batch_with_neighbourlist = {k:v.to(device) if type(v) is torch.Tensor else v for k,v in batch_with_neighbourlist.items()}
        return batch_with_neighbourlist
    
    # manually add selected edges to batch
    idx_i = []
    idx_j = []
    for a,b in connectivity:
        idx_i += [a,b]
        idx_j += [b,a]
    batch['_idx_i'] = torch.tensor(idx_i, device=device)
    batch['_idx_j'] = torch.tensor(idx_j, device=device)
    batch['_offsets'] = torch.zeros((len(idx_i), 3), device=device)
    batch['_idx_m'] = torch.zeros((n_atoms_total,), dtype=int, device=device)
    
    print('num edges:', len(batch['_idx_j'])/2)
    return batch