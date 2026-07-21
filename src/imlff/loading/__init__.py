import os
from schnetpack.atomistic.response import Forces
import torch
import wandb
from functools import lru_cache

from schnetpack.task import AtomisticTask
from imlff.common.util import proj_path
from imlff.model.deq import DEQModel

from imlff.model.iterator import AtomisticDEQ, AtomisticIterator
from imlff.model.solvers import fixed_point_iteration
from imlff.model.util import ForcesAndDeriveGBar


def turn_iter_into_deq(nnpot):
    """
    Turn an iterator into a DEQ model.
    """
    assert isinstance(nnpot.representation, AtomisticIterator), 'this function only works for AtomisticDEQ'
    n_iter = len(nnpot.representation.stack_schnets)
    stack_schnet = nnpot.representation.stack_schnets[0]
    for stack_schnet_2 in nnpot.representation.stack_schnets:
        assert stack_schnet == stack_schnet_2, 'all Schnets in the iterator must be the same'
    deq = DEQModel(solver=fixed_point_iteration, forw_max_iter=n_iter, back_max_iter=n_iter)
    iter_schnet = AtomisticDEQ(stack_schnet, deq)
    nnpot.representation = iter_schnet
    nnpot.collect_outputs()

def adapt_old_models(nnpot, turn_into_deq=False):

    cname = nnpot.representation.__class__.__name__
    if cname == 'AtomisticIterator':
        if turn_into_deq:
            turn_iter_into_deq(nnpot)
        else:
            return nnpot

    # just to be sure: copy. This adds some new attributes to old deq objects.
    nnpot.representation.deq = nnpot.representation.deq.__copy__()

    # if one object in nnpot.output_modules is of Forces, replace it with ForcesAndDeriveGBar
    for i, output_module in enumerate(nnpot.output_modules):
        if isinstance(output_module, Forces):
            nnpot.output_modules[i] = ForcesAndDeriveGBar()

    return nnpot


def load_model(model_path, map_location=None, turn_into_deq=False):
    nnpot = torch.load(model_path, map_location=map_location)
    nnpot = adapt_old_models(nnpot, turn_into_deq=turn_into_deq)
    nnpot.collect_outputs()
    return nnpot


def restore_checkpoint(ckpt_path, map_location=None, turn_into_deq=False, as_deq_task=False):
    print(ckpt_path)
    if as_deq_task:
        from imlff.training.task import AtomisticDEQTask
        TaskCls = AtomisticDEQTask
    else:
        TaskCls = AtomisticTask
    task = TaskCls.load_from_checkpoint(ckpt_path, map_location=map_location)
    task.model = adapt_old_models(task.model, turn_into_deq=turn_into_deq)
    task.model.collect_outputs()
    return task

@lru_cache(maxsize=2000)
def artifact_or_run_cached(model_tag, run=False, projects=('qcml-group/dev-v10', 'qcml-group/dev-v10-foo'), return_tag=False, verbose=True, **kwargs):
    print(f"Loading {model_tag} from wandb")
    api = wandb.Api()
    if '/' in model_tag:
        try:
            obj = api.run(model_tag, **kwargs) if run else api.artifact(model_tag, **kwargs)
        except Exception as e:  # incl. wandb CommError for in-flight/missing versions on running runs
            raise ValueError(('Run' if run else 'Artifact') + f" {model_tag} not found: {e}") from e
        if return_tag:
            return model_tag
        return obj
        
    for project in projects:
        try:
            obj_tag = f'{project}/{model_tag}'
            obj = api.run(obj_tag, **kwargs) if run else api.artifact(obj_tag, **kwargs)
            if return_tag:
                return obj_tag
            return obj
        except Exception as e:
            print(('Run' if run else 'Artifact') + f" {model_tag} not found in {project}: {e}")
    raise ValueError(('Run' if run else 'Artifact') + f" {model_tag} not found in any of the projects: {projects}")

def download_ckpt_from_wandb(artifact_name, verbose=True):
    path = proj_path('models', artifact_name)
    if not os.path.exists(path + '/model.ckpt'):
        artifact = artifact_or_run_cached(model_tag=artifact_name, type='model', verbose=verbose)
        artifact.download(root=path)
    return path + '/model.ckpt'

def load_model_from_wandb(artifact_name, map_location=None, turn_into_deq=False, return_ckpt=False, verbose=True, as_deq_task=False):
    ckpt_path = download_ckpt_from_wandb(artifact_name, verbose=verbose)
    ckpt = restore_checkpoint(ckpt_path, map_location=map_location, turn_into_deq=turn_into_deq, as_deq_task=as_deq_task)
    if return_ckpt:
        return ckpt
    return ckpt.model