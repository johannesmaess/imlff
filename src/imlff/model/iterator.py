from schnetpack import properties
from schnetpack.representation import SchNet, PaiNN, SO3net
import torch
import torch.nn as nn

class AtomisticDEQ(nn.Module):
    """
    This is a simple yet very flexible DEQ model.
    It consists of a function (usually an atomic representation in our case), and a DEQ solver.
    The solver is used to find a fixed point of the function during inference.
    """
    
    def __init__(self, f, deq):
        nn.Module.__init__(self)
        
        self.f = f
        self.deq = deq
        self.zemb_as_z0_default = False
        
    def get_z_shape(self, inputs):
        n_z = self.f.n_z
        bsz = len(inputs['_positions'])
        cn = self.f.AtomisticRepresentationClass.__name__
        if cn == 'PaiNN':
            return (bsz, 4, n_z)
        if cn == 'SchNet':
            return (bsz, n_z)
        if cn == 'SO3net':
            return (bsz, (self.f.lmax+1)**2, n_z)
        raise NotImplementedError(f"Unknown representation class {self.f.AtomisticRepresentationClass}")
        
    
    def forward(self, inputs):
        inputs['z_shape'] = self.get_z_shape(inputs)

        if getattr(self, 'zemb_as_z0_default', False) and 'z0' not in inputs:
            zemb = self.f.embed(inputs, None)

            # apply norm
            assert not self.f.norm_s or self.f.norm_c, 'Todo: Probably we have to do nothing here, but but you can think what might be optimal.'
            if self.f.norm_o:
                zemb = self.f.norm(zemb)

            inputs['z0'] = zemb

        # implicitly solve for the fixed point
        f_x = lambda z: self.f(inputs, z)
        z_star = self.deq(f_x=f_x, inputs=inputs)

        # save the fixed point as a representation
        self.f.save(inputs, z_star)
        return inputs
    
    
class AtomisticIterator(nn.Module):
    """
    Chained call to Schnet models, to get a comparison to AtomisticDEQ.
    Varying degree of weight tying of the interactions and embeddings.
    We iterate over AtomisticRepresentationStackable 
        (instead of iterating over multiple SchNetInteraction like Schnet does)
        to provide the embedded atom numbers at every iteration.
    """
    def __init__(self, stack_schnets):
        nn.Module.__init__(self)
        self.stack_schnets = nn.ModuleList(stack_schnets)
        
    def forward(self, inputs):
        z = None
        
        # compute the representation by iterating through the (potentially tied schnets)
        for stack_schnet in self.stack_schnets:
            z = stack_schnet.forward(inputs, z)
            
        # save the representation
        return self.stack_schnets[0].save(inputs, z)
