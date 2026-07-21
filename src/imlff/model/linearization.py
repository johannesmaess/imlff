      
from schnetpack.atomistic.distances import PairwiseDistances
import torch
from torch.autograd.functional import _as_tuple, _grad_preprocess, _check_requires_grad, _validate_v, _fill_in_zeros, _grad_postprocess, _tuple_postprocess, _autograd_grad

# the normal functorch jvp does not work, hence i use the following code:
# https://discuss.pytorch.org/t/combining-functional-jvp-with-a-nn-module/81215/6
def fw_linearize(func, inputs, create_graph=False, strict=False):
    is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jvp")
    inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)


    outputs = func(*inputs)
    is_outputs_tuple, outputs = _as_tuple(outputs, "outputs of the user-provided function", "jvp")
    _check_requires_grad(outputs, "outputs", strict=strict)
    # The backward is linear so the value of grad_outputs is not important as
    # it won't appear in the double backward graph. We only need to ensure that
    # it does not contain inf or nan.
    grad_outputs = tuple(torch.zeros_like(out, requires_grad=True) for out in outputs)

    grad_inputs = _autograd_grad(outputs, inputs, grad_outputs, create_graph=True)
    _check_requires_grad(grad_inputs, "grad_inputs", strict=strict)

    def lin_fn(v, retain_graph=True):
        if v is not None:
            _, v = _as_tuple(v, "v", "jvp")
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            _validate_v(v, inputs, is_inputs_tuple)
        else:
            if len(inputs) != 1 or inputs[0].nelement() != 1:
                raise RuntimeError("The vector v can only be None if the input to "
                                   "the user-provided function is a single Tensor "
                                   "with a single element.")

        grad_res = _autograd_grad(grad_inputs, grad_outputs, v, create_graph=create_graph, retain_graph=retain_graph)

        jvp = _fill_in_zeros(grad_res, outputs, strict, create_graph, "back_trick")

        # Cleanup objects and return them to the user
        jvp = _grad_postprocess(jvp, create_graph)

        return _tuple_postprocess(jvp, is_outputs_tuple)
    return lin_fn

# just to demonstrate the core functionality of fw_linearize (omits formatting and checks)
def fw_linearize_basic(func, inputs, create_graph=False):
    # inputs should be a tuple of tensors (same shapes as later in v)
    inputs = tuple(inp.detach().requires_grad_() for inp in inputs)
    output = func(*inputs)
    
    # how much had the inputs to change for a 0 change in output?
    # - creates a graph towards this quantity, such that it can be seen as a function of the '0' change in output
    grad_output = torch.zeros_like(output, requires_grad=True)
    grad_inputs = torch.autograd.grad(output, inputs, grad_output, create_graph=True)
    
    def lin_fn(v, retain_graph=True):
        # given the v change in (grad-)output, how much had the inputs to change?
        grad_res, = torch.autograd.grad(grad_inputs, grad_output, v, create_graph=create_graph, retain_graph=retain_graph)
        return grad_res
    
    return lin_fn


def fw_linearize_spk(layer, inputs_dict, R, z_star, create_graph=False, strict=False):
    def func(R, z_star):
        inputs_dict_copy = inputs_dict.copy()
        inputs_dict_copy['_positions'] = R
        inputs_dict_copy = PairwiseDistances.forward(None, inputs_dict_copy)
        return layer(inputs_dict_copy, z_star)
    return func, fw_linearize(func, (R, z_star), create_graph=create_graph, strict=strict)