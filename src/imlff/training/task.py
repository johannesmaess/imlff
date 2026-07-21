from __future__ import annotations
import copy

import torch
from imlff.loading import turn_iter_into_deq
import wandb

from ase import Atoms

import schnetpack as spk
import schnetpack as spk
from schnetpack.atomistic.distances import PairwiseDistances
from schnetpack.md.simulation_hooks.basic_hooks import SimulationHook
from schnetpack.md.simulator import Simulator
from schnetpack.md.initial_conditions import UniformInit

from torchdeq.loss.jacobian import jac_reg

from imlff.md.util import warmstart_config_str
from imlff.model.solvers import fixed_point_iteration, gather_max_within_molecule
from imlff.md.util import setup_md

def filter_indices(available_indices, requested_indices):
    available_indices = sorted(list(set(available_indices)))
    if requested_indices==True:
        return available_indices
    
    result = [i for i in available_indices if i in requested_indices]
    if -1 in requested_indices:
        idx = available_indices[-1]
        if idx not in result:
            result.append(idx)
    return result

def first_below_threshold(d, threshold):
    get_full = lambda val: torch.full_like(input=d[max(d)], fill_value=val, dtype=int)
    idx_first_below = get_full(max(d))
    for k in sorted(d)[-2::-1]: # reversed, all but the largest.
        is_below_threshold = d[k] <= threshold
        idx_first_below[is_below_threshold] = k
    return idx_first_below

class AtomisticTaskSafe(spk.task.AtomisticTask):
    def __init__(self, *args, turn_into_deq=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._new = True
        self.turn_into_deq = turn_into_deq
        
    def on_train_epoch_start(self):
        self.log('lr', self.lr_schedulers().get_last_lr()[0])
        if self._new:
            self.log('val_loss', torch.nan)
            if self.turn_into_deq:
                turn_iter_into_deq(self.model)
            self._new = False
    
    def on_train_epoch_end(self):
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                print()
                print('*'*50)
                print(f"NaN detected in {name}")
                print('*'*50)
                self.trainer.should_stop = True
                break
                # raise Exception(f"NaN detected in {name}")
    
class AtomisticDEQTask(AtomisticTaskSafe):
    def __init__(self, *args, 
                ## every loss term is parametrized by 
                # - which epoch it kicks it (set negative to deactivate) 
                # - a strength (weighting it multiplicatively)
                # - possibly other terms (explained below)
                
                # jacobian regularization
                jac_reg=-1,
                jac_reg_strength=1e-3, 
                
                # fit E&F to truncated prediction
                trunc=-1,
                trunc_strength=1e-3,
                trunc_at=[1,2],        # at which iterations to truncate and fit E&F
                trunc_gamma=1.,       # exponential discount factor for past fixpoint correction losses

                # fixed point correction
                fpc=-1,
                fpc_strength=1e-3,
                fpc_at=True,        # at which iterations to apply fixpoint correction. (pass a list  or None->never, True->always)
                fpc_gamma=1.,       # exponential discount factor for past fixpoint correction losses

                md_steps=-1,
                
                **kwargs):
        """
        Initializes the AtomisticDEQTask with an additional argument for Jacobian regularization.

        Args:
            Every loss term is parametrized by which epoch it kicks it (set negative to deactivate) and a strength (weighting it multiplicatively).
            *args: Variable length argument list for the parent class.
            **kwargs: Arbitrary keyword arguments for the parent class.
        """
        super().__init__(*args, **kwargs) 
        
        self.automatic_optimization = False
        
        self.jac_reg = jac_reg 
        self.jac_reg_strength = jac_reg_strength 
        self.fpc = fpc
        self.fpc_strength = fpc_strength
        self.fpc_at = fpc_at if fpc_at is not None else []
        self.fpc_gamma = fpc_gamma
        self.trunc = trunc
        self.trunc_strength = trunc_strength
        self.trunc_at = trunc_at if trunc_at is not None else []
        self.trunc_gamma = trunc_gamma

        # which residual errors to log:
        # (note that 0 denotes the residual of the 1st iteration)
        self.log_res_at = [0, 1, 2, 3, -1]
        self.log_res_at = True
        
        # at which iteration to calculate and log MAE of an intermediate E&F prediction
        # (1 denotes the result of the first iteration)
        self.log_metrics_at = [1, 2, 3]
        
        self.log_everything_in_val = True
        
        self.md_steps = md_steps

        self.forw_tol_for_logging = self.back_tol_for_logging = [0.0032, .0056, .01, .018, .032, .056, .1]

    def _exp(self, n, k, gamma):
        return gamma ** (n-k-1)
    
    def should_run(self, start_from_epoch):
        # if start_from_epoch is negative, the loss term is deactivated.
        return self.current_epoch >= start_from_epoch >= 0
    
    def default_log(self, log_dict, **overwrite_kwargs):
        log_dict = {k:v for k,v in log_dict.items() if v is not None}
        if not log_dict: return
        
        # check if we are in training or not (and confirm for every key)
        is_train = list(log_dict.keys())[0].startswith('train_')
        prefix = 'train_' if is_train else 'val_'
        for key in log_dict:
            assert key.startswith(prefix), f"key {key} does not start with {prefix}"
            
        kwargs = dict(
            on_step=is_train and False,
            on_epoch=True,
            batch_size=self.current_batch_size,
        )
        kwargs.update(overwrite_kwargs)
        
        self.log_dict(log_dict, **kwargs) # TODO logging still has troubles in old learning routing.
    
    def log_intermediate_metrics_and_loss_fpc(self, pred, targets, step_label, run_fpc, run_trunc, batch):
        ### Fixpoint correction loss
        ### and
        ### Trunc loss
        ### (intermediate-step metric logging removed to avoid val_metrics spam)

        loss = 0

        if 'zts' not in pred:
            return loss

        for t, zt in pred['zts'].items():
            if (zt is None) or (not zt.requires_grad) or (t==0):
                # zt must be attached to the computation graph
                # to calculate gradients wrt. model weights.
                continue

            fpc_at_this_t = self.should_run(self.fpc) \
                and (self.fpc_at is True or t in self.fpc_at)
            trunc_at_this_t = self.should_run(self.trunc) \
                and (self.trunc_at is True or t in self.trunc_at)

            if run_fpc and fpc_at_this_t:
                fpc_discount_t = self._exp(pred['forw_iterations'], t, self.fpc_gamma)
                z_star = pred['z_star']
                mse_fpc = torch.nn.functional.mse_loss(zt, z_star)
                # TODO: MAE only used in especially tagged runs so far. If successful, control with parameter:
                loss += self.fpc_strength * fpc_discount_t * mse_fpc

            if run_trunc and trunc_at_this_t and targets and batch:
                # perform no further deq iteration, just call the output modules with zt.
                pred_t, _ = self.modified_forward_step(batch=batch, z0=zt, forw_max_iter=0, back_max_iter=None, capture_back=False)
                # note: we are backpropagating...
                # - implicit_diff==True: pseudo-gradients with one attached iteration like the fixpoint correction paper.
                # - implicit_diff=False: backpropagating through the unrolled fixed_point_iteration solver.
                loss_t = self.loss_fn(pred_t, targets)
                trunc_discount_t = self._exp(pred['forw_iterations'], t, self.trunc_gamma)
                loss += self.trunc_strength * trunc_discount_t * loss_t
        return loss
    
    def loss_jac_reg(self, z_star, z_pre, step_label):
        if not self.should_run(self.jac_reg):
            return 0
        
        # compute jac reg loss
        in_train = 'train' == step_label
        reg_loss = jac_reg(z_star, z_pre, create_graph=in_train)
        
        self.default_log(log_dict={f"{step_label}_jac_reg_loss": reg_loss.item()})
        return self.jac_reg_strength * reg_loss
    
    def loss_normal(self, pred, targets, step_label):
        loss = self.loss_fn(pred, targets)
        self.log_metrics(pred, targets, step_label)
        self.default_log(log_dict={f"{step_label}_loss_normal": loss.item()})
        return loss
    
    def log_iterations_and_res(self, pred, step_label, idx_m):
        ### log iteration count & residual error in forward and backward iterative process
        log_dict = dict()
        
        if fi:=pred.get('forw_iterations'):
            log_dict[f"{step_label}_forw_iter"] = fi
            
        if bi:=pred.get('back_iterations'):
            log_dict[f"{step_label}_back_iter"] = bi
            
        if fr:=pred.get('forw_res'):
            # get the first iteration where the atoms residual goes below the threshold, compute max over molecule, mean over batch
            for tol in self.forw_tol_for_logging:
                fbt = first_below_threshold(fr, tol)
                fbt = gather_max_within_molecule(fbt, idx_m)
                fbt = fbt.mean(dtype=float).item()
                log_dict[f"{step_label}_forw_iter_max_{tol}"] = fbt   # maximum over molecule (only logged from version 11 forward)
            
            indices = filter_indices(fr.keys(), self.log_res_at)
            for t in indices:
                # get residual at t iterations, compute max over molecule, mean over batch
                res_t = gather_max_within_molecule(pred['forw_res'][t], idx_m)
                res_t = res_t.mean(dtype=float)
                log_dict[f"{step_label}_forw_res_{t:02d}"] = res_t.item()       # (still logged, but calculation changed with version 11)
                log_dict[f"{step_label}_forw_res_max_{t:02d}"] = res_t.item()   # maximum over molecule (only logged from version 11 forward)
        
        if br:=pred.get('back_res'):
            # get the first iteration where the atoms residual goes below the threshold, compute max over molecule, mean over batch
            for tol in self.back_tol_for_logging:
                fbt = first_below_threshold(br, tol)
                fbt = gather_max_within_molecule(fbt, idx_m)
                fbt = fbt.mean(dtype=float).item()
                log_dict[f"{step_label}_back_iter_{tol}"] = fbt       # (still logged, but calculation changed with version 11)
                log_dict[f"{step_label}_back_iter_max_{tol}"] = fbt   # maximum over molecule (only logged from version 11 forward)
                
            indices = filter_indices(br.keys(), self.log_res_at)
            for t in indices:
                # get residual at t iterations, compute max over molecule, mean over batch
                res_t = gather_max_within_molecule(pred['back_res'][t], idx_m)
                res_t = res_t.mean(dtype=float)
                log_dict[f"{step_label}_back_res_{t:02d}"] = res_t.item()       # (still logged, but calculation changed with version 11)
                log_dict[f"{step_label}_back_res_max_{t:02d}"] = res_t.item()   # maximum over molecule (only logged from version 11 forward)
                
        self.default_log(log_dict=log_dict)

    def modified_forward_step(self, batch=None, z0=None, capture_back=True, **modifiers):
        """
        Maintains the integrity of the batch dict by always only passing a shallow copy into the model.
        Can accept additional kwargs to overwrite the batch dict.
        Note: If values of the dict are modified inplace, the original batch dict will be modified.
        """
        batch = copy.copy(batch)
        if capture_back==False:
            batch.update(dict(capture_g=False, capture_g_bar=False, return_back_res=False, return_gts=False))
        batch.update(modifiers)
        if z0 is not None: # we clone (not detach) such that hooks on the z0's are not shared
            batch['z0'] = z0.clone()
        return self.forward_step(batch)
    
    def return_zts_for_fpc_and_trunc_and_logging(self):
        if self.log_metrics_at or (self.fpc_at and self.should_run(self.fpc)):
            if self.fpc_at is True or self.trunc_at is True:
                return True
            return sorted(set(self.log_metrics_at) | set(self.fpc_at) | set(self.trunc_at))
        return False

    def step_with_losses(self, batch, batch_idx, step_label):
        """
        Note: Best not write to batch dict, because it is used often.
        """
        # TODO
        # - Now that we are running with implicit diff: Currently in next-MD fast prediction, we allow many backward passes.

        
        #### FORWARD (basic) ####
        
        pred, targets = self.modified_forward_step(batch=batch, return_zts=self.return_zts_for_fpc_and_trunc_and_logging())
        z_star = pred['z_star']
        z_pre = pred['z_pre'] # one iteration before z_star
        
        # log iteration count & residual error in forward and backward iterative process
        self.log_iterations_and_res(pred, step_label, idx_m=batch['_idx_m'])
        
        #### LOSSES ####
        
        loss = 0 
        loss_normal = self.loss_normal(pred, targets, step_label)
        losses_normal = [loss_normal]
            
        # Intermediate metrics and fixpoint correction
        loss += self.log_intermediate_metrics_and_loss_fpc(pred=pred, targets=targets, step_label=step_label, batch=batch, run_fpc=True, run_trunc=True)

        if step_label == 'train':                      # log only in validation
            loss += sum(losses_normal) / len(losses_normal)

        return loss, z_star, z_pre

    def manual_optimizer_step(self, loss, retain_graph=True):
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss, retain_graph=retain_graph)
        optimizer.step()

    def manual_optimizer_step_split(self, loss, z_star, z_pre, step_label):
        """DESIGN IDEA (not currently in use — see problem below):

        Intended to reduce peak memory by sequencing the two second-order graphs:
          1. Backprop force loss (retain_graph=False) → forward graph freed.
          2. Build jac_reg second-order graph on the now-freed-and-cheap z_pre→z_star subgraph.
          3. Backprop jac_reg (retain_graph=False) → second-order graph freed.

        CURRENT PROBLEM:
        Step 1 cannot use retain_graph=False because jac_reg still needs the z_pre→z_star
        subgraph for its VJP in step 2.  So in practice retain_graph=True is required for
        step 1, which keeps the entire forward graph (R → z1 → ... → z_star → E) alive in
        GPU memory.  The second manual_backward frees only the jac_reg second-order graph,
        not the original retained forward graph — which then bleeds into the next training
        step's forward pass and causes OOM.

        HOW IT COULD WORK IN THE FUTURE:
        If forces (= -dE/dR) could be computed *after* jac_reg rather than during the
        SchNetPack forward pass, the sequence becomes viable:
          1. Forward pass produces z_star, z_pre (no force computation yet).
          2. Compute jac_reg loss and backprop it (retain_graph=False) → frees forward graph.
          3. Re-run or checkpoint the z_star → E → F path separately and backprop force loss.
        This would keep only one second-order graph in memory at a time.
        """
        optimizer = self.optimizers()
        optimizer.zero_grad()
        needs_jac_reg = step_label == 'train' and self.should_run(self.jac_reg)
        self.manual_backward(loss, retain_graph=needs_jac_reg)
        if needs_jac_reg:
            jac_loss = self.loss_jac_reg(z_star=z_star, z_pre=z_pre, step_label=step_label)
            self.manual_backward(jac_loss, retain_graph=False)
        optimizer.step()
        
    def get_md_steps(self, step_label):
        if type(self.md_steps) is tuple:
            steps_train, steps_val = self.md_steps
            return steps_train if step_label == 'train' else steps_val
        return self.md_steps
            
    def calc_loss_and_manual_step(self, batch, batch_idx, step_label):
        class_name = self.model.representation.__class__.__name__
        assert class_name == 'AtomisticDEQ', 'use other task'
        self.current_batch_size = len(batch["_idx"])
        
        md_steps = self.get_md_steps(step_label)
        
        if md_steps < 0: # old training, no training-time MD
            loss, z_star, z_pre = self.step_with_losses(batch, batch_idx, step_label)
            loss = loss + self.loss_jac_reg(z_star=z_star, z_pre=z_pre, step_label=step_label)
            if not self.automatic_optimization and step_label == 'train':
                self.manual_optimizer_step(loss)
        else: # new, with training-time MD.
            
            # TODO:
            # - Add gradient space losses

            # to be captured in OptimizationHook class:
            task = self
            
            # define hook which can update weights after every simulation step
            class OptimizationHook(SimulationHook):
                def __init__(self):
                    super().__init__()
                    self._simulator = None
                    self.before_simulation_start = True
                    self._current_step_label = step_label
                    self.loss_per_step = []
                
                def _set_current_step_label(self):
                    # in the very first force calculation (before the simulator loop) no warmstarting is possible, nothing needs to be marked.
                    if self.before_simulation_start:
                        self._current_step_label = step_label
                        return
                    
                    step = self._simulator.step + 1 # count of how often the position had been updated.
                    warmstart_str = warmstart_config_str(**self._simulator.calculator.get_warmstart_configs_available())
                    self._current_step_label = f"{step_label}_s{step:03d}_{warmstart_str}"
                    
                def _loss_fpc_and_trunc(self):
                    # fpc/trunc losses only apply at the initial coldstart step.
                    run_fpc = self.before_simulation_start
                    run_trunc = self.before_simulation_start

                    # todo: resolve this issue
                    if step_label == 'val': run_trunc = False
                    assert not run_trunc, "md train implementation does not support calculating forces from intermediate z yet.\nnon-implicit diff has the problem that md simulator feed new position tensors (even if positions didn't change), thus prohibiting differentiation.\nimplicit diff might require more changes."
                    
                    results = self._simulator.calculator.results
                    tar, bat = None, None
                    if self.before_simulation_start and run_trunc:
                        _, tar = task.apply_constraints(results, batch.copy())
                        tar = task.filter_for_outputs(tar)
                        bat = batch
                    return task.log_intermediate_metrics_and_loss_fpc(pred=results, targets=tar, batch=bat, run_fpc=run_fpc, run_trunc=run_trunc, step_label=self._current_step_label)
                
                def _loss_jac_reg(self):
                    z_star = self._simulator.calculator.results['z_star']
                    z_pre = self._simulator.calculator.results['z_pre'] # one iteration before z_star
                    return task.loss_jac_reg(z_star=z_star, z_pre=z_pre, step_label=self._current_step_label)
                
                def _loss_normal(self):
                    results = self._simulator.calculator.results
                    pred, targets = task.filter_for_outputs(results), task.filter_for_outputs(batch)
                    pred, targets = task.apply_constraints(pred, targets)
                    loss = task.loss_normal(pred=pred, targets=targets, step_label=self._current_step_label)
                    return loss
                
                def _info(self, loss):
                    results = self._simulator.calculator.results
                    task.log_iterations_and_res(pred=results, step_label=self._current_step_label, idx_m=batch['_idx_m'])
                    
                    return
                    prefix = 'S' if self.before_simulation_start else self._simulator.step
                    fi = results['forw_iterations']
                    fr = [f"{k}: {v.item():.5f}" for k,v in results['forw_res'].items()]
                    print(prefix, fi, loss.item())#, fr)
                    
                def _step(self, loss):
                    if step_label == 'train' and type(loss) is torch.Tensor:
                        task.manual_optimizer_step(loss)
                    self.loss_per_step.append(loss.clone().detach() if type(loss) is torch.Tensor else loss)
                    self._info(loss)
                    
                def on_simulation_start(self, simulator: Simulator):
                    # this is called after the initial forward pass
                    self._simulator = simulator
                    
                    loss = self._loss_normal()
                    loss += self._loss_fpc_and_trunc()
                    loss += self._loss_jac_reg()
                    self._step(loss)
                    
                    self.before_simulation_start = False
                    
                def on_step_begin(self, simulator: Simulator):
                    # the step label has to be updated before the start of the step,
                    # because it relies on the length of the available hisotory *before* the step.
                    self._set_current_step_label()
                    
                def on_step_end(self, simulator: Simulator):
                    loss = 0
                    loss += self._loss_fpc_and_trunc()
                    loss += self._loss_jac_reg()
                    self._step(loss)
            
            warmstart_degree=1
            md_config = dict(
                timestep=0.5,
                temperature=500,
                # langevin_time_constant=1000,
                initial_temperature=500, # TODO: check that these changes don't break training
                Init=UniformInit
            )
            warmstart_configs=dict(z=warmstart_degree, g_bar=warmstart_degree) if warmstart_degree>=0 else None
            
            num = batch['_atomic_numbers'].cpu().numpy()
            pos = batch['_positions'].cpu().numpy()
            n_atoms = batch['_n_atoms']
            pivot, molecules = 0, []
            for n in n_atoms:
                molecules.append(Atoms(numbers=num[pivot:pivot+n], positions=pos[pivot:pivot+n]))
                pivot += n
                
            md_simulator, _ = setup_md( # TODO: check that these changes don't break training
                    model=self.model, 
                    atoms=molecules,
                    md_config=md_config,
                    warmstart_configs=warmstart_configs,
                    log_properties=[],
                    hooks=[oh:=OptimizationHook()],
                    progress=False,
                    device=batch["_positions"].device,
                    )
            if 'train' in step_label:
                # revert model.eval() call in init of md_calculator
                self.model.train()
            
            deq = self.model.representation.deq
            deq._tmp_config = dict(
                implicit_diff = deq.implicit_diff or step_label=='val',
                capture_g_bar=True,
                return_zts=self.return_zts_for_fpc_and_trunc_and_logging(),
                return_gts=self.return_zts_for_fpc_and_trunc_and_logging(),
            )
            
            md_simulator.simulate(md_steps)
        
            # revert modifications to model
            self.model.train()
            self.model.representation.deq._tmp_config = None
            
            loss = sum(oh.loss_per_step) if oh.loss_per_step else None
        
        self.current_batch_size = None
        log_dict = {step_label+'_loss': loss}
        self.default_log(log_dict=log_dict, prog_bar=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        print('', end='', flush=True) # just to get stuff out
        self.model.train()
        return self.calc_loss_and_manual_step(batch, batch_idx, 'train')
    
    def on_validation_epoch_start(self):
        self.val_losses = []
        
    def remember_val_loss(self, val_loss):
        self.val_losses.append(val_loss)
    
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        torch.set_grad_enabled(self.grad_enabled) # otherwise no_grad, and forces can not be calculated.
        loss_scalar = loss = self.calc_loss_and_manual_step(batch, batch_idx, 'val')
        if type(loss_scalar) is dict:
            loss_scalar = loss_scalar['val_loss']
        self.remember_val_loss(torch.tensor(loss_scalar, dtype=float) if type(loss_scalar) in (float, int) else loss_scalar.detach())
        return loss

    def on_validation_epoch_end(self):
        AtomisticTaskSafe.on_validation_epoch_end(self)
        if self.val_losses:
            val_loss = torch.stack(self.val_losses).mean(dtype=float)
            scheduler = self.lr_schedulers()
            if scheduler is not None:
                scheduler.step(val_loss)
        
class AtomisticDEQTaskAutomaticOptimization(AtomisticDEQTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.get_md_steps('train') == -1, "automatic optimization is not possible for training-time MD."
        self.automatic_optimization = True
        
    def manual_optimizer_step(self, loss, retain_graph=True):
        pass
        
    def remember_val_loss(self, val_loss):
        pass