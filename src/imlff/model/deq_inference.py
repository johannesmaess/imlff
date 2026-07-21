import torch
import torch.nn as nn
import torch.autograd as autograd

from imlff.common.util import collect_config
from imlff.model.deq import DEQModel, IterationWrap
from imlff.model.solvers import fixed_point_iteration


class DEQInferenceFixedPointSolve(torch.autograd.Function):
    """
    torch.autograd.Function implementing the DEQ forward solve and implicit
    differentiation backward pass for inference (no training, no back-back).
    """

    @staticmethod
    def forward(ctx, R, z0, g_bar0, f_x, config, batch_size, inputs):
        assert config.backward
        assert config.implicit_diff
        assert config.capture_g_bar
        assert config.iterate_on_g_bar
        assert not config.capture_g
        assert config.forw_max_iter > 0, f"forw_max_iter must be > 0, but got {config.forw_max_iter=}."
        assert config.back_max_iter >= 1, f"back_max_iter must be >= 1, but got {config.back_max_iter=}."
        
        def F(z):
            with torch.enable_grad():
                return f_x(z)

        forw_iteration = IterationWrap(
            F,
            initial_value=z0,
            write_dict=inputs,
            capture_intermediates=config.return_zts,
            intermediates_key='zts',
            residual_errors_key='forw_res',
            counter_key='forw_iterations',
        )

        z_final_m1, z_final = config.forw_solver(
            forw_iteration, z0,
            max_iter=config.forw_max_iter, tol=config.forw_tol,
            batch_size=batch_size, reattach_before_iterations=config.implicit_diff
        )

        assert z_final_m1 is not None, "Todo: Generalize this to the case where z0 is not passed and we need just one iteration."

        # Save z_final_m1 (leaf) and z_final (f_x grad_fn) directly.
        # Return z_final.clone() — PyTorch only replaces the clone's grad_fn with the
        # Function node, leaving the saved original's f_x grad_fn intact for backward.
        ctx.save_for_backward(R, z_final_m1, z_final)
        ctx.g_bar0 = g_bar0
        ctx.config = config
        ctx.batch_size = batch_size
        ctx.inputs = inputs

        with torch.enable_grad():
            return z_final.clone() # return a clone bc. torchs grad_fn modification would bring us into recursion

    @staticmethod
    def backward(ctx, grad):
        R, z_final_m1, z_final = ctx.saved_tensors
        g_bar0 = ctx.g_bar0
        config = ctx.config
        batch_size = ctx.batch_size
        inputs = ctx.inputs
        
        # avoid the circular reference memory leak.
        # Note: still leaking if autograd is never called. (Use weakref then)
        ctx.inputs = None

        def B(g_bar_prev):  # one backward iteration
            if config.verbose:
                print('back iter (on g_bar)')
            g = g_bar_prev + grad
            g_bar, dR = autograd.grad(z_final, (z_final_m1, R), g, retain_graph=True, create_graph=False)
            with torch.no_grad():
                if torch.isnan(g_bar).any():
                    print('Warning: g_bar contains NaNs.')
            B.dR = dR # the last one that we save will be used for force prediction
            return g_bar

        if g_bar0 is None:
            g_bar0 = grad.clone().detach()

        back_iteration = IterationWrap(
            B,
            initial_value=g_bar0,
            write_dict=inputs,
            capture_intermediates=config.return_gts,
            intermediates_key='g_barts',
            residual_errors_key='back_res',
            counter_key='back_iterations',
        )

        _, g_bar_star = config.back_solver(back_iteration, g_bar0,
            max_iter=config.back_max_iter, tol=config.back_tol,
            batch_size=batch_size, reattach_before_iterations=False)
        dEdR = B.dR
        del B # avoid potential memory leaks through circular references

        # Note: g_bar_star could be passed out as some derivative, and picked up in ForcesAndDeriveGBar,
        # but we have 'inputs' here anyways to write the back_iteration stats.
        inputs['g_bar_star'] = g_bar_star 
        # g_star = g_bar_star + grad

        # Return dE/dR (= J_{f,R}^T @ g_star) as R's gradient.
        return dEdR, None, None, None, None, None, None


class DEQModelInference(DEQModel):
    """
    Lightweight Torch autograd function based implementation
    """

    def forward(self, f_x, inputs):
        """_summary_

        Args:
            f_x (function): self.f(..., x) i.e. the implicit layer, which is already partially applied with x.
            z0 (torch.Tensor): initial guess for z.
            g_bar0 (torch.Tensor): initial guess for g_bar.
        """
        R = inputs['_positions']
        assert R.requires_grad
        assert not self.training

        config = self.collect_class_config(inputs)
        assert config.implicit_diff

        if config.verbose:
            print('DEQModelInference Running ', id(inputs))

        z0 = inputs.get('z0')
        g0 = inputs.get('g0')
        g_bar0 = inputs.get('g_bar0')
        if z0 is not None:
            z0.requires_grad_()

        batch_size = len(inputs["_idx"]) if "_idx" in inputs else None
        z_final = DEQInferenceFixedPointSolve.apply(R, z0, g_bar0, f_x, config, batch_size, inputs)
        
        return z_final


def to_inference(model: DEQModel) -> DEQModelInference:
    inf = DEQModelInference(
        forw_solver=model.forw_solver,
        back_solver=model.back_solver,
        implicit_diff=True,
        forw_max_iter=model.forw_max_iter, forw_tol=model.forw_tol,
        back_max_iter=model.back_max_iter, back_tol=model.back_tol,
        verbose=getattr(model, 'verbose', False),
        iterate_on_g_bar=True,
        capture_g=False,
    )
    inf.eval()
    return inf
