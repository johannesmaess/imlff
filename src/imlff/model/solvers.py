from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn



def calc_diff(x1, x2):
    with torch.no_grad():
        def norm(x):
            return x.view(len(x), -1).norm(p=2, dim=-1)
        if x1 is None:
            return torch.nan * norm(x2)
        rel_diff = norm(x1 - x2) / norm(1e-5 + x1)
        return rel_diff
    
def gather_max_within_molecule(vals, idx_m):
    with torch.no_grad():
        k = max(idx_m) + 1
        result = torch.full((k,), float('-inf'), device=vals.device)
        result = result.scatter_reduce(0, idx_m, vals.to(dtype=result.dtype), reduce='amax', include_self=True)
        return result.to(dtype=vals.dtype)

def anderson(f, x0, m=2, lam=1e-4, max_iter=100, tol=1e-4, beta=1., 
             batch_size=None, reattach_before_iterations=True):
    """
    Anderson acceleration for fixed point iteration.
    Note: doesn't help our I-MLFF, good warmststarts make us converge before history is even builts up.
    """
    assert max_iter >= 0, f"{max_iter=} >=! 0"
    assert m >= 2, f"{m=} >=! 2 otherwise use fixed_point_iteration"
    assert batch_size is not None, f"{batch_size=}"
    
    if max_iter==0:
        print("Warning: max_iter=0, returning input, gradient graph uncontrolled.")
        return None, x0
    
    if isinstance(beta, (list, tuple)):
        betas = beta
        assert len(betas) >= max_iter, f"{len(betas)=} < {max_iter=}"
    else:
        betas = [beta]*(max_iter+1)
        
    # todo: beta is not used in first two iterations.
    
    x1 = f(x0)
    if max_iter==1 or calc_diff(x0, x1).max().item() < tol:
        return x0, x1
    
    # collect shapes for function call and anderson storage
    f_shape_batch = x1.shape
    n_elem_batch = torch.tensor(f_shape_batch).prod().item()
    n_elem = int(n_elem_batch / batch_size)
    
    m = min(m, max_iter)
    
    X = torch.zeros(batch_size, m, n_elem, dtype=x1.dtype, device=x1.device)
    F = torch.zeros(batch_size, m, n_elem, dtype=x1.dtype, device=x1.device)
    if x0 is not None: # (else it was interpreted by f as zero, so we don't have to change anything)
        X[:,0] = x0.view(batch_size, n_elem)
    F[:,0] = x1.view(batch_size, n_elem)
    
    H = torch.zeros(batch_size, m+1, m+1, dtype=x1.dtype, device=x1.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(m+1, dtype=x1.dtype, device=x1.device)
    y[0] = 1
    
    for k in range(1, max_iter):
        beta = betas[max(k, len(betas)-1)]
        assert 0<beta, f"{beta=}"
        
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x1.dtype,device=x1.device)[None]
        # Note: this can raise as linalg error: The solver failed because the input matrix is singular.
        alpha = torch.linalg.solve(H[:, :n+1, :n+1], y[None, :n+1, None])[:, 1:n+1, 0]   # (batch_size x n)
        X[:,k%m] \
            =    beta  * (alpha[:,None] @ F[:,:n])[:,0] \
            + (1-beta) * (alpha[:,None] @ X[:,:n])[:,0]
        x_last_m1 = X[:,k%m].reshape(f_shape_batch)
        if reattach_before_iterations:
            x_last_m1 = x_last_m1.detach().requires_grad_()
        x_last = f(x_last_m1)
        F[:,k%m] = x_last.detach().view(batch_size, n_elem)
        if (calc_diff(x_last_m1, x_last).max().item() < tol):
            break
        
    return x_last_m1, x_last

def fixed_point_iteration(f, x, max_iter=200, tol=1e-4, beta=1.,
                          batch_size=None, reattach_before_iterations=True):
    """
    Note: beta!=1 doesn't help us.
    """
    
    if isinstance(beta, (list, tuple)):
        betas = beta
    else:
        betas = [beta]*max_iter
        
    if max_iter==0:
        print("Warning: max_iter=0, returning input, gradient graph uncontrolled.")
        return None, x
        
    for i in range(max_iter):
        beta = betas[min(i, len(betas)-1)]
        # assert 0<beta<=1, f"{beta=}"
        assert 0<beta, f"{beta=}"
        
        if reattach_before_iterations and x is not None and i>0:
            x_last_m1 = x.detach().requires_grad_()
            # Drop every reference to the PREVIOUS iteration's autograd tape
            # before building this iteration's tape. Otherwise the prior tape
            # (still pinned by `x` via the mixing step and by the stale `x_last`
            # binding) coexists with the new one — two full f-graphs resident at
            # once = the ~2x peak. Under reattach the prior tape is never used
            # for gradients (the input above is already detached; only the final
            # iteration's tape is saved for the IFT backward), so dropping it is
            # numerically a no-op. (The IterationWrap's `last_value` is the third
            # holder — it detaches its own copy in remember_value.)
            x = x.detach()
            x_last = None
        else:
            x_last_m1 = x
        x_last = f(x_last_m1)
        diff = calc_diff(x, x_last).max().item()
        if (diff <= tol):
            break
        # self mixing:
        x = beta * x_last + (0 if x is None else (1-beta) * x)
    return x_last_m1, x_last
