import os
from datetime import datetime
import pytorch_lightning as pl

import schnetpack as spk
from imlff.training.callbacks import CleanWandbArtifactsCallback, TimeLimitCallback, CustomBestMetricTracker
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def validate(task, datamodule, model_dir, **trainer_kwargs):
    trainer = pl.Trainer(
        default_root_dir=model_dir,
        max_epochs=0,
        limit_val_batches=1.0,
        **trainer_kwargs        
    )
    return trainer.validate(task, datamodule=datamodule)


def train(task, datamodule, logger=None, deadline=None, ckpt_path=None, early_stopping=True, retry_ckpt_path=None, **trainer_kwargs):
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
    print(f"Starting training at {time_str}")
    # EarlyStopping patience counts *validation runs*, not epochs. We want to
    # terminate after ~600 EPOCHS of no val_loss decrease regardless of how
    # often we validate, so divide the target epoch budget by
    # check_val_every_n_epoch (default 1 if unset/None).
    _cve = trainer_kwargs.get("check_val_every_n_epoch") or 1
    _es_patience = max(1, round(600 / _cve))
    print(f"EarlyStopping: ~600 epochs no-decrease -> patience={_es_patience} "
          f"val-checks (check_val_every_n_epoch={_cve})")
    # Create callbacks to sav best model, track metrics and stop training if deadline is reached
    callbacks = [
        spk.train.ModelCheckpoint(
            save_last=True,
            every_n_epochs=100,
            model_path=f"/tmp/best_{time_str}",
            save_top_k=1,
            monitor="val_loss",
            mode='min',
        ),
        metric_tracker_loss:=CustomBestMetricTracker(monitor="val_loss"),
        CustomBestMetricTracker(monitor="val_forces_MAE"),
    ] + early_stopping * [
        EarlyStopping(monitor="val_loss", min_delta=0.00, patience=_es_patience, verbose=False, mode="min", check_finite=False),
    ]
    if logger is not None:
        callbacks.append(CleanWandbArtifactsCallback())
    if deadline is not None:
        callbacks.append(TimeLimitCallback(deadline))
    
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        default_root_dir="/tmp",
        **trainer_kwargs
    )
    if ckpt_path:
        print('starting from checkpoint', ckpt_path)
    print(f'{metric_tracker_loss.last_epoch=}')   
    try:
        trainer.fit(task, datamodule=datamodule, ckpt_path=ckpt_path)
    except KeyboardInterrupt as e:
        print(f'{metric_tracker_loss.last_epoch=}')
        raise e
    except Exception:
        if retry_ckpt_path is not None:
            try:
                trainer.save_checkpoint(retry_ckpt_path)
                print(f'Saved retry checkpoint to {retry_ckpt_path} (epoch={metric_tracker_loss.last_epoch})')
            except Exception as save_err:
                print(f'Failed to save retry checkpoint: {save_err}')
        raise

    return metric_tracker_loss.last_epoch
