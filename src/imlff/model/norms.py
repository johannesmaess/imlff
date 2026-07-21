import torch
import torch.nn as nn

class InvUnitLengthFeaturesNorm(nn.Module):
    tag = 'InvUnitL2Feat'
    def __init__(self, n_z=None, eps=1e-5):
        super(InvUnitLengthFeaturesNorm, self).__init__()
        self.eps = eps
    
    def forward(self, input):
        assert input.ndim == 2
        r = input.norm(dim=-1, p=2, keepdim=True).clip(min=1e-9, max=None)
        return input / (self.eps + r)

    
class EquiUnitLengthFeaturesNorm(nn.Module):
    tag = 'EquiUnitL2Feat'
    def __init__(self, n_z=None, eps=1e-5):
        super(EquiUnitLengthFeaturesNorm, self).__init__()
        self.eps = eps
    
    def forward(self, input):
        assert input.ndim == 3
        r = input.norm(dim=(1,2), p=2, keepdim=True).clip(min=1e-9, max=None)
        return input / (self.eps + r)

## Merged norm ##
# - receives both invariant,equivariant. shape (n_atoms, 1+3+5+..., feature size)

class EquiMergedLayerNorm(nn.Module):
    """
    Equivariant Merged Layer Normalization (EquiformerV3, Liao et al. 2024).

    Normalizes an irreps feature with a single shared RMS σ computed across
    all spherical-harmonic degrees L in [0, Lmax]. Only the scalar part
    (L=0) has its channel mean subtracted and a bias β added; higher
    degrees receive a per-degree gain γ⁽ᴸ⁾ only.

    Per-atom statistics:
        μ⁽⁰⁾ = mean_i x⁽⁰⁾_{0,i}
        σ⁽⁰⁾² = mean_i (x⁽⁰⁾_{0,i} - μ⁽⁰⁾)²
        σ⁽ᴸ⁾² = mean_{m,i} (x⁽ᴸ⁾_{m,i})²          for L > 0
        σ²   = mean_L σ⁽ᴸ⁾²

    Outputs:
        y⁽⁰⁾ = γ⁽⁰⁾ ⊙ (x⁽⁰⁾ - μ⁽⁰⁾) / σ + β⁽⁰⁾
        y⁽ᴸ⁾ = γ⁽ᴸ⁾ ⊙ x⁽ᴸ⁾ / σ                   for L > 0

    Input shape: (n_atoms, (Lmax+1)², C), with degrees concatenated along
    dim 1 in ascending order (L=0 at index 0, L=1 at 1..3, L=2 at 4..8, ...).
    """
    tag = 'EquiMergedLN'

    def __init__(self, n_z, lmax, eps=1e-5):
        super().__init__()
        self.C = int(n_z)
        self.lmax = int(lmax)
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(self.lmax + 1, self.C))
        self.beta_0 = nn.Parameter(torch.zeros(self.C))

    def forward(self, x):
        # Accept 2D (N, C) — treat as scalar-only, Lmax=0 — for SchNet-style
        # invariant inputs, and return a tensor of the same rank as the input.
        input_ndim = x.ndim
        if input_ndim == 2:
            assert self.lmax == 0
            x = x.unsqueeze(1)
        assert x.ndim == 3
        _, D, C = x.shape
        assert (self.lmax + 1) ** 2 == D
        assert C == self.C

        # σ⁽ᴸ⁾² per atom, per degree
        x0 = x[:, 0, :]                                  # (N, C)
        mu0 = x0.mean(dim=-1, keepdim=True)              # (N, 1)
        x0c = x0 - mu0                                   # (N, C)

        sigma_sq_per_L = [x0c.pow(2).mean(dim=-1)]       # [(N,)]
        for L in range(1, self.lmax + 1):
            start = L * L
            end = start + (2 * L + 1)
            xL = x[:, start:end, :]                      # (N, 2L+1, C)
            sigma_sq_per_L.append(xL.pow(2).mean(dim=(1, 2)))

        sigma = torch.stack(sigma_sq_per_L, dim=-1).mean(dim=-1)   # (N,)
        sigma = (sigma + self.eps).sqrt().view(-1, 1, 1)           # (N,1,1)

        out = torch.empty_like(x)
        out[:, 0:1, :] = self.gamma[0] * (x0c.unsqueeze(1) / sigma) + self.beta_0
        for L in range(1, self.lmax + 1):
            start = L * L
            end = start + (2 * L + 1)
            out[:, start:end, :] = self.gamma[L] * (x[:, start:end, :] / sigma)

        if input_ndim == 2:
            out = out.squeeze(1)
        return out
