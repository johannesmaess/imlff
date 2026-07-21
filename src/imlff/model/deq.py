from functools import partial
import torch
import torch.nn as nn
import torch.autograd as autograd

from imlff.common.util import collect_config
from imlff.model.solvers import calc_diff, fixed_point_iteration

class IterationWrap():
    """
    Wrap a function to be iterated to count calls and optionally capture intermediate results and residual errors.
    You can pass a dict to write to, or let IterationWrap create one to read it out later.
    """
    
    def __init__(
        self,
        func,
        initial_value,
        write_dict=None,
        capture_intermediates=None,
        counter_key=None,
        intermediates_key=None,
        residual_errors_key=None,
        ):
        """
        Args:
            func (function): Function to be wrapped and iterated.
            initial_value (Optional[Tensor]): Initial value before first iteration of function.
            write_dict (dict, optional): You can pass a dict to write to, or let IterationWrap create one to read it out later.
            capture_intermediates (Union[List, Boolean, None], optional): List of ints for iterations to capture. Pass True to capture all. Pass None to capture None. Defaults to None.
            intermediates_key (str, optional): _description_. Key under which intermediates will be written into write_dict.
            residual_errors_key (str, optional): _description_. Key under which residuals will be written into write_dict.
        """
        
        self.func = func
        self.write_dict = write_dict if write_dict is not None else dict()
        
        self.counter_key = counter_key
        self.counter = 0
        
        self.capture_intermediates = capture_intermediates if capture_intermediates is not None else []
        self.intermediates_key = intermediates_key
        self.residual_errors_key = residual_errors_key
        
        self.remember_value(initial_value)
        
        
    def __call__(self, inp):
        out = self.func(inp)
        self.counter += 1
        self.remember_res(out)
        self.remember_value(out)
        return out
    
    # define counter getter with python attributes
    @property
    def counter(self):
        if self.counter_key is None:
            return self._counter
        else:
            return self.write_dict[self.counter_key]
    
    @counter.setter
    def counter(self, value):
        if self.counter_key is None:
            self._counter = value
        else:
            self.write_dict[self.counter_key] = value
            
    # increase counter by one:
    def increase_counter(self):
        self.counter += 1
    
    
    def remember_res(self, value):
        """
        Compares and stores the residual error compared to the last seen value.
        If the residual_errors_key is set.
        """
        if self.residual_errors_key is None:
            return
        
        if self.residual_errors_key not in self.write_dict:
            self.write_dict[self.residual_errors_key] = dict()
        
        # calculate and write value    
        res = calc_diff(self.last_value, value)
        self.write_dict[self.residual_errors_key][self.counter] = res
        
    
    def remember_value(self, value):
        """
        Stores the value at the current counter,
        If the counter is in the capture_intermediates list, and the intermediates_key is set.
        """
        # Store a DETACHED copy as last_value: it is only ever read by
        # remember_res (a no_grad residual comparison), so it does not need the
        # autograd tape — and keeping the tape here pins the previous
        # iteration's f-graph across the next iteration's forward pass, which is
        # one of the three references behind the ~2x DEQ-inference memory peak
        # (see fixed_point_iteration). The captured-intermediates path below
        # keeps the un-detached value (zts analysis may want the graph).
        self.last_value = value.detach() if torch.is_tensor(value) else value
        
        if self.intermediates_key is None \
            or self.capture_intermediates is None \
            or self.capture_intermediates is False:
            return
        assert self.capture_intermediates is True or isinstance(self.capture_intermediates, list), f"{type(self.capture_intermediates)=}"
        
        if self.capture_intermediates is True or self.counter in self.capture_intermediates:
            if self.intermediates_key not in self.write_dict:
                self.write_dict[self.intermediates_key] = dict()

            # write value
            self.write_dict[self.intermediates_key][self.counter] = value
        
class DEQModel(nn.Module):
    """
    Note: The models forward call does not detach z0.
    This means that it's the users choice if gradients should flow to it's creation,
    e.g. a previous MD-steps output.
    """
    
    model_outputs = ['z_star', 'g_star', 'g_bar_star', 'z_pre', 'zts', 'gts', 'g_barts', 'forw_res', 'back_res', 'forw_iterations', 'back_iterations']
    
    def __init__(self, 
                 solver=None, forw_solver=None, back_solver=None,
                 implicit_diff=True, 
                 forw_max_iter=10, forw_tol=1e-4,
                 back_max_iter=10, back_tol=1e-4,
                 iterate_on_g_bar=True, capture_g=False,
                 verbose=False
                 ):
        super().__init__()
        self.implicit_diff = implicit_diff
        
        self.forw_solver = forw_solver or solver
        self.back_solver = back_solver or solver
        assert self.forw_solver is not None
        assert self.back_solver is not None
        
        self.forw_max_iter = forw_max_iter
        self.back_max_iter = back_max_iter
        self.forw_tol = forw_tol
        self.back_tol = back_tol
        
        self.iterate_on_g_bar = iterate_on_g_bar
        self.capture_g = capture_g
        
        self.verbose = verbose
        
        self._last_inputs = None
        
    def get_tmp_config(self):
        if hasattr(self, '_tmp_config'):
            return self._tmp_config or dict()
        return dict()
        
    def collect_class_config(self, override_config):
        """
        Specifies which config elements DEQModel uses, as well as some defaults that are other than None.
        """
        config_elements = [
            'implicit_diff',
            'forw_solver', 'forw_max_iter', 'forw_tol',
            'back_solver', 'back_max_iter', 'back_tol',
            # Set return_zts/return_gts to True to return all intermediate zts/gts, to a list of integers to return only those iterations, or to None to return none.
            # Set return_forw_res/return_back_res to False to return no residual errors.
            'return_zts', 'return_forw_res',
            'return_gts', 'return_back_res',
            'backward',
            'capture_g', 'capture_g_bar', 'capture_back_stats', 'iterate_on_g_bar',
            'verbose',
        ]
        defaults = dict(
            return_forw_res=True, backward=True, return_back_res=True, capture_g=True, capture_g_bar=True, iterate_on_g_bar=False,
            )
        return collect_config(override_config, self.get_tmp_config(), self, defaults, config_elements=config_elements)
    
    def forward(self, f_x, inputs):
        """_summary_

        Args:
            f_x (function): self.f(..., x) i.e. the implicit layer, which is already partially applied with x.
            z0 (torch.Tensor): initial guess for z.
            g0 (torch.Tensor): initial guess for g.
            g_bar0 (torch.Tensor): initial guess for g_bar.
        """
        
        config = self.collect_class_config(inputs)
        batch_size = len(inputs["_idx"]) if "_idx" in inputs else None
        
        if config.verbose:
            print('Running ', id(inputs))
        
        z0 = inputs.get('z0')
        if z0 is not None:
            z0.requires_grad_()
        elif not config.implicit_diff:
            assert 'z_shape' in inputs, 'we need to know which shape to initialize z0 with if not implicit_diff and z0 is not passed.'
            z0 = torch.zeros(inputs['z_shape'], requires_grad=True, device=inputs['_positions'].device)
        g0 = inputs.get('g0')
        if g0 is not None:
            print('g0 is not None')
        g_bar0 = inputs.get('g_bar0')
        
        assert config.forw_max_iter >= 0, f"forw_max_iter must be >= 0, but got {config.forw_max_iter=}."
        if config.forw_max_iter == 0:
            # we perform no further computation
            assert z0.requires_grad, "if forw_max_iter == 0, z0 must already have it's dependencies on model weights and inputs in place."
            assert config.back_max_iter in ['?', None], "if forw_max_iter == 0, we competely rely on it's defined backward pass. acknowledge that."
            return z0
        
        # at least one backward pass will always be performed:
        # if forw_max_iter >= 1, this will be defined in z_final.
        # if forw_max_iter == 0, this was defined in z0, which is the z_final of the previous MD step.
        # more might be in the IFT backward solver, or unrolled forw_solver.
        assert config.back_max_iter >= 1, f"back_max_iter must be >= 1, but got {config.back_max_iter=}."
        
        assert not (self.training and config.implicit_diff), 'train using unrolled forward solver'
        
        forw_iteration = IterationWrap(
            f_x, 
            initial_value=z0,
            write_dict=inputs,
            capture_intermediates=config.return_zts,
            intermediates_key='zts',
            residual_errors_key='forw_res',
            counter_key='forw_iterations',
        )
        
        ## compute forward pass
        z_final_m1, z_final = config.forw_solver(
            forw_iteration, z0,
            max_iter=config.forw_max_iter, tol=config.forw_tol,
            batch_size=batch_size, reattach_before_iterations=config.implicit_diff
        )
        
        # save last iteration for jacobian regularisation:
        inputs['z_pre'] = z_final_m1
        
        if not config.backward: # don't remove this, as future if conditions depend on it.
            return z_final
        
        # prepare backward pass
        if config.capture_g_bar or (config.implicit_diff and (config.capture_g or config.return_back_res or config.return_gts)):
            # store reference to inputs dict for backward hook:
            self._last_inputs = inputs
            if config.verbose:
                print('set last inputs:', id(inputs))
        elif config.verbose:
            print('not storing inputs:', id(inputs))
        
            
        # attach hook to z_final_m1 to capture the g_bar in the backward pass
        def capture_g_bar_hook(g_bar_final, check_inputs_id):
            if config.verbose: print('capture_g_bar_hook')
            if self._last_inputs == 'energy dervative done':
                if config.verbose:
                    print('energy dervative done. not saving derivative as we assume this is part of a loss calculation etc')
                return
            
            assert check_inputs_id == id(self._last_inputs), f"{check_inputs_id}!={id(self._last_inputs)} inputs dict is not the same as when capture_g_bar_hook was set up."
            self._last_inputs['g_bar_star'] = g_bar_final
            with torch.no_grad():
                if g_bar_final.isnan().any():
                    print('Warning: g_bar_final contains NaNs.')
            # clear reference to inputs dict:
            self._last_inputs = 'energy dervative done'
        
        if not config.implicit_diff:
            if config.capture_g_bar:
                # todo: later the gradient is computed successfully,
                # but i never check if this 'g_bar' makes sense.
                # (why would we capture g_bar when not running implicit diff?)
                z0.register_hook(partial(capture_g_bar_hook, check_inputs_id=id(inputs)))
                inputs['derive_for_g_bar'] = z0
            return z_final
                
        assert z_final_m1 is not None, "Todo: Generalize this to the case where z0 is not passed."
        # z_final_m1 = torch.zeros_like(z0)
        assert z_final_m1.requires_grad, "z_final_m1 must have requires_grad=True."
        assert z_final.requires_grad, "z_final should have requires_grad=True, because it is created from z_final_m1."
        if config.capture_g_bar:
            inputs['derive_for_g_bar'] = z_final_m1
            z_final_m1.register_hook(partial(capture_g_bar_hook, check_inputs_id=id(inputs)))
        
        ## prepare backward pass
        # compute another detached forward pass. only to set up the vjp g*J_f(z_final)
        # todo: this can in principle be done with autograd.grad(z_final, z_final_m1,...) directly. but this leads to a weird torch timeout.
        z_final_m1_detached = z_final_m1.clone().detach().requires_grad_() if z_final_m1 is not None else torch.zeros_like(z_final).requires_grad_() # init if no iterations were performed, else detach.
        z_final_detached = f_x(z_final_m1_detached) # don't use forw_iteration, we don't want to capture detached intermediates.

        if config.verbose:
            print('implicit diff:', config.implicit_diff)

        # set up Jacobian vector product (without additional forward calls)
        def back_hook(grad, g0, g_bar0, check_inputs_id):
            ### If we are performing force calculation: grad=dE/dz_final.
            write_to = None
            if self._last_inputs == 'energy dervative done':
                if config.verbose:
                    print('energy dervative done. not saving backward results as we assume this is part of a loss calculation etc')
            elif config.capture_g or config.return_gts or config.return_back_res:
                assert check_inputs_id == id(self._last_inputs), "inputs dict is not the same as when back_hook was set up." # todo: one case can still easily happen: when capture_g=False, self._last_inputs is not updated. but the back_iteration Wrapper still tries to write to it. So keep this check!
                write_to = self._last_inputs

            if config.iterate_on_g_bar:
                ## Determine starting value
                # 1. g_bar is the preferred option as it gives us one more iteration for free:
                # (roughly... the jvp was performed on the last conformation, the addition on the current.)
                if g_bar0 is not None:
                    pass
                # 2. otherwise start with the previous iterations last g guess (now called g0).
                elif g0 is not None:
                    assert False, 'This is not implemented, bc I wont use it. But I think you can start with: g_bar0=g0-grad'
                # 3. Maybe the user also decides on a constant initialization:
                elif type(g_bar0) == int or type(g_bar0) == float:
                    g_bar0 = torch.full_like(grad, g_bar0)
                # 4. if neither is passed, initialize with the additive component of the current conformation:
                elif g_bar0 is None:
                    g_bar0 = grad.clone().detach()
                else:
                    assert False, 'g_bar0 should be of type int, float or None'
            
                def B(g_bar_prev): # one backward iteration
                    if config.verbose: print('back iter (on g_bar)')

                    g = g_bar_prev + grad
                    g_bar, = autograd.grad(z_final_detached, z_final_m1_detached, g, retain_graph=True)
                    with torch.no_grad():
                        if torch.isnan(g_bar).any():
                            print('Warning: g_bar contains NaNs.')
                    return g_bar
                
                intermediates_key='g_barts'
                back0 = g_bar0
                
            else: # Iterate on g
                ## Determine starting value
                # 1. g_bar is the preferred option as it gives us one more iteration for free:
                # (roughly... the jvp was performed on the last conformation, the addition on the current.)
                if g_bar0 is not None:
                    g0 = g_bar0 + grad
                # 2. otherwise start with the previous iterations last g guess (now called g0).
                # 3. Maybe the user also decides on a constant initialization:
                elif type(g0) == int or type(g0) == float:
                    g0 = torch.full_like(grad, g0)
                # 4. if neither is passed, initialize with the additive component of the current conformation:
                elif g0 is None:
                    g0 = grad.clone().detach()
                else:
                    assert False, 'g0 should be of type int, float or None'
            
                def B(g): # one backward iteration
                    if config.verbose: print('back iter')
                    g_bar, = autograd.grad(z_final_detached, z_final_m1_detached, g, retain_graph=True)
                    with torch.no_grad():
                        if torch.isnan(g_bar).any():
                            print('Warning: g_bar contains NaNs.')
                    return g_bar + grad
                
                intermediates_key='gts'
                back0 = g0
                
            back_iteration = B if not write_to else IterationWrap(
                B, 
                initial_value=back0,
                write_dict=write_to,
                capture_intermediates=config.return_gts,
                intermediates_key=intermediates_key,
                residual_errors_key='back_res',
                counter_key='back_iterations',
            )
            
            # emulate a cleverer infrastructure. (can be found in deq_inference.py)
            # we actually perform one more backward iteration than we should, 
            # but then return the 2nd last g.
            # this allows the solver to stop by tolerance already after the first iteration.
            if config.verbose:
                print('back solver 1. tol=', config.back_tol)
            back_final_m1, _ = config.back_solver(back_iteration, back0,
                max_iter=config.back_max_iter, tol=config.back_tol,
                batch_size=batch_size, reattach_before_iterations=False)
            if config.iterate_on_g_bar:
                g_star = back_final_m1 + grad
            else:
                g_star = back_final_m1
                
            if write_to and config.capture_g:
                write_to['g_star'] = g_star
            if not config.capture_g_bar:
                # this is the last hook - tag _last_inputs as done.
                self._last_inputs = 'energy dervative done'
            return g_star
                
        z_final.register_hook(partial(back_hook, g0=g0, g_bar0=g_bar0, check_inputs_id=id(inputs)))
        return z_final
    
    def __copy__(self):
        return self.__class__( # Todo: Resolve own class dynamically
            solver=None, forw_solver=self.forw_solver, back_solver=self.back_solver,
            implicit_diff=self.implicit_diff,
            forw_max_iter=self.forw_max_iter, forw_tol=self.forw_tol,
            back_max_iter=self.back_max_iter, back_tol=self.back_tol,
            
            # because of backward compatibility (we are not sure that these are set):
            back_back_solver=getattr(self, 'back_back_solver', self.forw_solver),
            back_back_max_iter=getattr(self, 'back_back_max_iter', self.forw_max_iter),
            back_back_tol=getattr(self, 'back_back_tol', self.forw_tol),
            iterate_on_g_bar=getattr(self, 'iterate_on_g_bar', True),
            capture_g=getattr(self, 'capture_g', False),
            verbose=getattr(self, 'verbose', False),
        )
    
    def __deepcopy__(self, memo):
        return self.__copy__()
