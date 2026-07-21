from typing import Dict, Optional

import torch
import torch.nn as nn

from imlff.common.util import collect_config
from imlff.model.norms import EquiMergedLayerNorm

"""
A note on embed() in the follwowing child classes of AtomisticRepresentation
---
Usually embed's task is to embed the atomic numbers into a feature space.
But DeqLayer not only processes the input but also the recurrent state.

Thus we generalize and implement embed to combine the atom embeddings with the recursive state in various ways.
For now, both of their shapes are the same (n_atom_basis), but this might be generalized in the future.
"""

class AtomisticRepresentationStackable(nn.Module):
    """
    This class enables to execute atomistic representations (like SchNet or PaiNN) sequentially.
    At each step, it embeds the atom numbers and adds those embeddings to the recursive state.
    This class can be used for creating DEQ models by iterating until the recursive state reaches a fix-point.
    """
    def __init__(self,
                 AtomisticRepresentationClass: nn.Module,
                 n_z: int,
                 *args,
                 InvNormClass: nn.Module | None = None,
                 EquiNormClass: nn.Module | None = None,
                 MergedNormClass: nn.Module | None = None,
                 norm_s = False,
                 
                 norm_c = False,
                 norm_o = True, 
                 verbose = False,
                 **kwargs):

        kwargs.setdefault('n_atom_basis', n_z) # (default)
        self.AtomisticRepresentationClass = AtomisticRepresentationClass
        self.AtomisticRepresentationClass.__init__(self, *args, **kwargs)
        
        self.register_buffer('n_z', torch.tensor(n_z))
        
        
        if MergedNormClass is not None:
            n = AtomisticRepresentationClass.__name__
            if n=='SchNet':
                lmax = 0
            elif n=='PaiNN':
                lmax = 1
            else:
                lmax = self.lmax
            self.norm_merged = MergedNormClass(n_z=self.n_z, lmax=lmax)
        else:
            if InvNormClass is not None:
                self.norm_inv = InvNormClass(n_z=self.n_z)
            if EquiNormClass is not None:
                self.norm_equi = EquiNormClass(n_z=self.n_z)
            
        self.verbose = verbose
        self.norm_s = norm_s     # normalize z at start of iteration, before being combined with the atomic embeddings
        self.norm_c = norm_c     # normalize after combining with the atom embedding
        self.norm_o = norm_o     # normalize the output
    
    def norm(self, z):
        if hasattr(self, 'norm_merged'):
            return self.norm_merged(z)
        
        if not hasattr(self, 'norm_inv') and not hasattr(self, 'norm_equi'):
            return z
        
        norm_inv  = getattr(self, 'norm_inv',  lambda x: x)
        if z.ndim==2: # SchNet
            return norm_inv(z)
        
        norm_equi = getattr(self, 'norm_equi', lambda x: x)
        q, mu = torch.split(z, [1, z.shape[1]-1], dim=1)
        z_normed = torch.empty_like(z)
        z_normed[:,  0] = norm_inv(q.squeeze(1))
        z_normed[:, 1:] = norm_equi(mu)
        return z_normed
        
    def forward(self, inputs: Dict[str, torch.Tensor], z: Optional[torch.Tensor]):
        config = collect_config(inputs, self, config_elements=['verbose'])
        if config.verbose: print(f"forw")

        if getattr(self, 'norm_s', False) and z is not None:
            z = self.norm(z)

        # combine recurrect state and individual atom information
        z_emb = self.embed(inputs, z)

        if getattr(self, 'norm_c', False):
            z_emb = self.norm(z_emb)

        # compute interaction blocks. geom is geometry-derived precompute that
        # could be hoisted out of the recurrence (positions don't change between
        # iterations); for now we recompute it every call to preserve semantics.
        geom = self.AtomisticRepresentationClass.geom_embed(self, inputs)
        z_int = self.AtomisticRepresentationClass.interact(self, geom, z_emb)

        # readout (trims only if emb and z were concatenated before)
        z_readout = z_int
        if hasattr(self, 'n_z') and self.n_atom_basis > self.n_z:
            z_readout = z_int[:, :self.n_z]

        if getattr(self, 'norm_o', True):
            z_readout = self.norm(z_readout)

        return z_readout

    def embed(self, inputs: Dict[str, torch.Tensor], z: torch.Tensor | None):
        """
        Return a combination of atom embeddings and the recursive state.
        """
        combined = self.AtomisticRepresentationClass.atom_embed(self, inputs)
        if z is not None:
            assert z.shape == combined.shape, f"Expected {combined.shape}, got {z.shape}"
            combined += z
        return combined
    
    def save(self, inputs: Dict[str, torch.Tensor], z_star: torch.Tensor):
        """
        Save the current state.
        """
        inputs['z_star'] = z_star
        return self.AtomisticRepresentationClass.save(self, inputs, z_star)
    