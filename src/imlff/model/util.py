import copy
from typing import Dict, List
from schnetpack import properties
import torch
import torch.nn as nn
import torch.nn.functional as F
import schnetpack as spk
import schnetpack.transform as trn

from schnetpack.datasets import MD17
from schnetpack.representation import PaiNN, SO3net
from imlff.model.deq import DEQModel
from imlff.model.deq_inference import to_inference
from imlff.model.norms import EquiUnitLengthFeaturesNorm, InvUnitLengthFeaturesNorm, EquiMergedLayerNorm
from imlff.model.solvers import anderson, fixed_point_iteration
from imlff.model.representation import AtomisticRepresentationStackable
from imlff.model.iterator import AtomisticDEQ, AtomisticIterator


class ForcesAndDeriveGBar(spk.atomistic.Forces):
    def get_required_derivatives(self, inputs: Dict[str, torch.Tensor]) -> List[str]:
        required_derivatives = copy.copy(super().get_required_derivatives(inputs))
        if 'derive_for_g_bar' in inputs:
            # required_derivatives might have two entries already (for forces and stress). 
            # insert at position 1 to not interfere with them (force indexes 0, stress indexes -1):
            required_derivatives.insert(1, 'derive_for_g_bar')
        return required_derivatives
    def forward(self, inputs):
        if not inputs.get('backward', True):
            return inputs
        return super().forward(inputs)
    
class ZeroEmbedding(nn.Module):
    def forward(self, *_, **__):
        return 0
    
def get_schnets_for_iterator(n_schnets, shared_schnets=False, embed_in_later_schnets=True, **schnet_kwargs):
    schnets = [AtomisticRepresentationStackable(**schnet_kwargs)]
    
    # they all share the same embedding and interactions
    if shared_schnets: return n_schnets * schnets
    
    # they share the same nuclear embedding
    embedding = schnets[0].embedding if embed_in_later_schnets else ZeroEmbedding()
    return schnets + [AtomisticRepresentationStackable(nuclear_embedding=embedding, **schnet_kwargs) for _ in range(1, n_schnets)]

def init_iterating_schnet(device=None,
                    # Schnet/PaiNN (and their inner interactIONS)
                    n_interactions=1,
                    shared_interactions=False,
                    shared_filters=False,
                    n_z=128,
                    n_rbf=200,
                    cutoff=20.,
                    activation = F.silu,
                    lmax=2,
                    
                    # AtomisticRepresentationStackable
                    AtomisticRepresentationClass=PaiNN,
                    InvNormClass=True,
                    EquiNormClass=True,
                    MergedNormClass=None,
                    norm_s=False,
                    norm_c=False,
                    norm_o=True,
                    
                    # AtomisticDEQ or AtomisticIterator
                    use_deq=True,
                    
                    # AtomisticDEQ
                    implicit_diff=True,
                    forw_solver=fixed_point_iteration,
                    forw_tol=0,
                    forw_max_iter=10,
                    back_solver=fixed_point_iteration,
                    back_tol=0,
                    back_max_iter=10,
                    deq_model_inference=False,

                    # AtomisticIterator
                    # - should have tied interact(...) functions?
                    # (which might themselve consist of multiple (un-)tied interactIONS)
                    n_schnets=1,
                    shared_schnets=False,
                    
                    get_key=False,
                    ):
    
    Rep = AtomisticRepresentationClass.__name__
    
    if not implicit_diff:
        if forw_solver is not fixed_point_iteration:
            print(f"Note: {forw_solver=} is overwritten to fixed_point_iteration by {implicit_diff=}.")
            forw_solver = fixed_point_iteration
        if back_solver is not fixed_point_iteration:
            print(f"Note: {back_solver=} is overwritten to fixed_point_iteration by {implicit_diff=}.")
            back_solver = fixed_point_iteration
    
    # Overwrites others:
    if MergedNormClass:
        # Merged norm normalizes the whole irreps tensor; separate Inv/Equi
        # norms would double-normalize the scalar part.
        InvNormClass = EquiNormClass = None
        
    # Set default:
    if InvNormClass is True:
        InvNormClass=InvUnitLengthFeaturesNorm
    if EquiNormClass is True:
        EquiNormClass=EquiUnitLengthFeaturesNorm
    if MergedNormClass is True:
        MergedNormClass = EquiMergedLayerNorm            
            
    if not get_key:
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
        radial_basis = spk.nn.GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)
        
        stack_schnet_kwargs = dict(
            n_z=n_z,
            n_interactions=n_interactions,
            radial_basis=radial_basis,
            activation=activation,
            shared_interactions=shared_interactions,
            cutoff_fn=spk.nn.CosineCutoff(cutoff),
            norm_s=norm_s,
            norm_c=norm_c,
            norm_o=norm_o,
            MergedNormClass=MergedNormClass,
            InvNormClass=InvNormClass,
            EquiNormClass=EquiNormClass,
            AtomisticRepresentationClass=AtomisticRepresentationClass
        )
            
        if Rep=='PaiNN':
            stack_schnet_kwargs['shared_filters'] = shared_filters
        if Rep=='SO3net':
            stack_schnet_kwargs['lmax'] = lmax
            
        if use_deq is False:
            # something that iterates over the same/multiple Schnets
            stack_schnets = get_schnets_for_iterator(n_schnets=n_schnets, shared_schnets=shared_schnets, **stack_schnet_kwargs)
            iter_schnet = AtomisticIterator(stack_schnets).to(device)
        else:
            stack_schnet = AtomisticRepresentationStackable(**stack_schnet_kwargs).to(device)
            if use_deq:
                # our DEQ implementation
                deq = DEQModel(
                    forw_solver=forw_solver, forw_tol=forw_tol, forw_max_iter=forw_max_iter,
                    back_solver=back_solver, back_tol=back_tol, back_max_iter=back_max_iter,
                    implicit_diff=implicit_diff
                )
                if deq_model_inference:
                    deq = to_inference(deq)
            iter_schnet = AtomisticDEQ(stack_schnet, deq).to(device)

        pred_energy = spk.atomistic.Atomwise(n_in=n_z, output_key=MD17.energy)
        pred_forces = ForcesAndDeriveGBar(energy_key=MD17.energy, force_key=MD17.forces)
        # pred_forces_nc = spk.atomistic.Atomwise(n_in=n_z, n_out=3, output_key='forces_nc', per_atom_output_key='force_nc', aggregation_mode=None) # TODO continue non-conservative forces (in loss function etc.)
        
        nnpot = spk.model.NeuralNetworkPotential(
            representation=iter_schnet,
            input_modules=[pairwise_distance],
            output_modules=[pred_energy, pred_forces],
            postprocessors=[
                # trn.CastTo64(), # unavailable on mps
                trn.CastTo32(),
                trn.AddOffsets(MD17.energy, add_mean=True, add_atomrefs=False)
            ]
        ).to(device)
        
        return nnpot, iter_schnet, stack_schnet if use_deq else stack_schnets
    
    else:
        # create the key for this model
        s = lambda x: 's' if x else ''
        
        if Rep == 'SchNet':
            repr_str = 'schnet'
        elif Rep == 'PaiNN':
            repr_str = 'painn'
        elif Rep == 'SO3net':
            repr_str = 'so3net'
        else:
            raise NotImplementedError(f"Unknown AtomisticRepresentationClass: {AtomisticRepresentationClass}")

        key = f"{repr_str}_{'deq' if use_deq else 'iter'}_i{n_interactions}{s(shared_interactions)}_b{n_z}"
        if n_rbf != 50 and cutoff != 5.:
            key += f'_rbf{n_rbf}_cut{cutoff}'
            
        norm_str = ''
        
        if MergedNormClass is not None:
            norm_str += '_'+MergedNormClass.tag
        else:
            if InvNormClass is not None:
                if hasattr(InvNormClass, 'tag'):
                    norm_str += '_'+InvNormClass.tag
                else:
                    assert InvNormClass is nn.BatchNorm1d, "Todo: implement keys for other NormClasss"
                    norm_str += "_normBN"
            if (EquiNormClass is not None) and (Rep != 'SchNet'):
                norm_str += '_'+EquiNormClass.tag
        if norm_str:
            norm_st_str = '_normAt'
            if norm_s: norm_st_str += 'S'
            if norm_c: norm_st_str += 'C'
            if norm_o: norm_st_str += 'O'
            if not norm_st_str in '_normAtO':
                key += norm_st_str + norm_str
        # ^^ add nothing to key, if norm is applied nowhere
        if use_deq: 
            key += "_" + solving_str(implicit_diff, forw_solver, back_solver, forw_tol, back_tol, forw_max_iter, back_max_iter)
        elif n_schnets>1:
            key += f"_ns{n_schnets}{s(shared_schnets)}"
        return key
    
def solving_str(implicit_diff, forw_solver, back_solver, forw_tol, back_tol, forw_max_iter, back_max_iter):
    solver_str = lambda solver: ('A' if solver == anderson else 'F') if implicit_diff else 'FNI' # FNI: fixed point no implicit differentiation
    
    key = f"solv{solver_str(forw_solver)}_tol{forw_tol}_mi{forw_max_iter}"
    if implicit_diff and (
        (forw_solver != back_solver)
        or (forw_tol != back_tol)
        or (forw_max_iter != back_max_iter)):
            key += f"_bsolv{solver_str(back_solver)}_btol{back_tol}_bmi{back_max_iter}"

    return key

def adapt_apply_state_dict(nnpot, state_dict, verbose=False):
    target_state_dict = nnpot.state_dict()
    
    # print non-intersecting keys
    if verbose: print("Keys who are in state_dict but not in nnpot:")
    for k in state_dict.keys():
        if k not in target_state_dict:
            if verbose: print(k)
    if verbose: print("Keys who are in nnpot but not in state_dict:")
    for k in target_state_dict.keys():
        if k not in state_dict:
            if verbose: print(k)

    # print which shapes don't match
    if verbose: print("Shapes that don't match:")
    for k in state_dict.keys():
        if k in target_state_dict:
            if state_dict[k].shape != target_state_dict[k].shape:
                if verbose: print(k, state_dict[k].shape, target_state_dict[k].shape)
                
                if (state_dict[k].shape==torch.Size([])) and (target_state_dict[k].shape == torch.Size([1])):
                    state_dict[k] = state_dict[k][None]
                else:
                    print(k, state_dict[k].shape, target_state_dict[k].shape, "NO RESHAPING RULE FOUND!")
    if verbose: print("---")
    
    nnpot.load_state_dict(state_dict)
    if verbose: print("Loaded state_dict into nnpot.")
