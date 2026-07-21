import torch
import torchmetrics
import schnetpack as spk

from schnetpack.datasets import MD17
from schnetpack.train.lr_scheduler import ReduceLROnPlateau
import wandb

from imlff.training.task import AtomisticDEQTask, AtomisticDEQTaskAutomaticOptimization, AtomisticTaskSafe

def setup_task(nnpot, forces_in_loss=0., old_task='auto', automatic_optimization=True,
               lr_patience_epochs=250., check_val_every_n_epoch=5, **reg_kwarg):
    assert 0 <= forces_in_loss <= 1
    # ReduceLROnPlateau steps on val_loss, so its patience is in *validation
    # runs*, not epochs. Target ~lr_patience_epochs EPOCHS of no decrease before
    # halving the LR, regardless of validation frequency:
    _lr_patience = max(1, round(lr_patience_epochs / (check_val_every_n_epoch or 1)))
    output_energy = spk.task.ModelOutput(
        name=MD17.energy,
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1-forces_in_loss,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )
    output_forces = spk.task.ModelOutput(
        name=MD17.forces,
        loss_fn=torch.nn.MSELoss(),
        loss_weight=forces_in_loss,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )
    
    class_name = nnpot.representation.__class__.__name__
    use_old_task = old_task==True or (class_name == 'AtomisticIterator' and old_task=='auto')
    if use_old_task:
        TaskClass = AtomisticTaskSafe
    else:
        assert class_name == 'AtomisticDEQ', 'unsupported model'
        if automatic_optimization:
            TaskClass = AtomisticDEQTaskAutomaticOptimization
        else:
            TaskClass = AtomisticDEQTask
    
    task = TaskClass(
        model=nnpot,
        outputs=[output_energy, output_forces],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-3},
        scheduler_cls=ReduceLROnPlateau,
        scheduler_monitor="val_loss",
        scheduler_args=
        {
            "factor": 0.5,
            "patience": _lr_patience,
            "smoothing_factor": 0.9
        },
        **reg_kwarg,
        )
    return task

def delete_artifacts(wandb_entity, wandb_name, collection_name, dry_run, save_mode):
    try:
        api = wandb.Api(overrides={"entity": wandb_entity, "project": wandb_name})
        project = api.project(wandb_name)

        for artifact_type in project.artifacts_types():
            if artifact_type.type == 'model':
                for artifact_collection in artifact_type.collections():
                    # if collection_name is passed, delete only artifacts from that collection:
                    if (collection_name is None) or (artifact_collection.name == collection_name):
                        artifacts = api.artifacts(type_name='model', name=artifact_collection.name)
                        if save_mode:
                            alias_exists = any(len(artifact.aliases) > 0 for artifact in artifacts)
                            if not alias_exists:
                                print(f'\033[38;5;214mNO ALIAS EXISTS FOR\033[0m {artifact_collection.name}, no artifacts were deleted for this collection')
                                return
                        for artifact in artifacts:
                            if len(artifact.aliases) > 0:
                                print(f'\033[32mKEEPING \033[0m {artifact.name}:{artifact.version}, aliases: {artifact.aliases}')
                            else:
                                print(f'\033[31mDELETING\033[0m {artifact.name}:{artifact.version}, aliases: {artifact.aliases}')
                                if not dry_run:
                                    artifact.delete()
    except Exception as e:
        print(f'\033[38;5;214mWARNING:\033[0m artifact cleanup failed ({type(e).__name__}: {e}); skipping.')
