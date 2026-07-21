import math
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.utilities import _scan_checkpoints
from torch import Tensor
from pathlib import Path
import wandb
from requests.exceptions import HTTPError
import time

import numpy as np

from imlff.loading.epoch_meta import read_ckpt_epoch_step


class WandBLoggerClearArtifacts(pl.loggers.WandbLogger):
    def __init__(self, keep_every_epochs=None, keep_every_seconds=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.keep_every_epochs = keep_every_epochs
        self.keep_every_seconds = keep_every_seconds

        self.tagged_last_at_epochs = 0
        self.tagged_last_at_seconds = 0

    def _val_metrics_from_summary(self, epoch) -> dict:
        """Validation metrics for `epoch`, formatted as run_val.py's annotation.

        run_val.py re-validates every artifact post-hoc and stamps the result
        into artifact.metadata as `{metric}_manual` keys. The exact same numbers
        are already produced by training's validation epochs and live in the
        wandb run summary, so we mirror that annotation at upload time — letting
        run_val.py's metadata cache reuse them instead of re-validating.

        We only stamp when the summary's `epoch` matches this checkpoint's epoch;
        otherwise the logged `val_*` metrics belong to a different model (e.g. a
        resume re-saving a stale last.ckpt). The metric cleaning mirrors
        run_val.py: drop `<x>_max` duplicates, keep only `val_forw_res_<NN>`
        among `_res_` keys.
        """
        try:
            summary = dict(self.experiment.summary.items())
        except Exception as e:
            print(f"could not read wandb summary for artifact val annotation: {e}")
            return {}

        summary_epoch = summary.get("epoch")
        if (epoch is not None and summary_epoch is not None
                and int(summary_epoch) != int(epoch)):
            return {}

        metrics = {}
        for k, v in summary.items():
            if not k.startswith("val_"):
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if math.isnan(v):
                continue
            metrics[k] = v

        # drop `<x>_max` only if `<x>` exists with the same value
        for m in list(metrics.keys()):
            if "_max" in m:
                n = m.replace("_max", "")
                if n in metrics and metrics[n] == metrics[m]:
                    metrics.pop(m)

        # keep only `val_forw_res_<NN>` among `_res_` keys
        for m in list(metrics.keys()):
            if "_res_" in m and m[:-2] != "val_forw_res_":
                metrics.pop(m)

        return {f"{k}_manual": v for k, v in metrics.items()}

    def _log_one_checkpoint(self, checkpoint_callback: ModelCheckpoint, t, p, s, _tag) -> None:
        """Single-checkpoint upload, copied from Lightning's WandbLogger but with
        authoritative `epoch` / `global_step` injected into artifact.metadata.

        We can't post-hoc-stamp after super()'s upload because there's no robust
        way to map a freshly-uploaded artifact-version back to the local ckpt
        path (especially across the resume-time race where multiple uploads can
        be in flight). Stamping at upload-time guarantees correctness.

        Authoritative epoch comes from torch.load on the local ckpt file. The
        `epoch=N-step=M.ckpt` filename template is NOT trusted: Lightning fills
        it from a template at save time and we've observed cases where the
        filename disagrees with the pickled `ckpt['epoch']` (e.g. resumed runs
        re-saving an old `last.ckpt` that retains a stale name).
        """
        filename = Path(p).name
        epoch, global_step = read_ckpt_epoch_step(p)

        metadata = {
            "score": s.item() if isinstance(s, Tensor) else s,
            "original_filename": filename,
            checkpoint_callback.__class__.__name__: {
                k: getattr(checkpoint_callback, k)
                for k in [
                    "monitor",
                    "mode",
                    "save_last",
                    "save_top_k",
                    "save_weights_only",
                    "_every_n_train_steps",
                ]
                if hasattr(checkpoint_callback, k)
            },
        }
        if epoch is not None:
            metadata["epoch"] = int(epoch)
        if global_step is not None:
            metadata["global_step"] = int(global_step)

        # Stamp this epoch's validation scores as `{metric}_manual`, matching
        # run_val.py's post-hoc annotation so its metadata cache can skip
        # re-validating this artifact.
        metadata.update(self._val_metrics_from_summary(epoch))

        if not self._checkpoint_name:
            self._checkpoint_name = f"model-{self.experiment.id}"
        artifact = wandb.Artifact(name=self._checkpoint_name, type="model", metadata=metadata)
        artifact.add_file(p, name="model.ckpt")
        aliases = ["latest", "best"] if p == checkpoint_callback.best_model_path else ["latest"]
        self.experiment.log_artifact(artifact, aliases=aliases)
        self._logged_model_time[p] = t

    # this logger only keeps the checkpoints which have an active alias in wandb.
    def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        # Replaces super()._scan_and_log_checkpoints so we can write the real
        # epoch into artifact.metadata at upload time (Lightning's upstream
        # implementation only records score + filename).
        for t, p, s, tag in _scan_checkpoints(checkpoint_callback, self._logged_model_time):
            self._log_one_checkpoint(checkpoint_callback, t, p, s, tag)
        
        if self.experiment.entity == 'dummy':
            # wandb_mode == 'disabled'
            return
        
        api = wandb.Api()
        key = self.experiment.entity + "/" + self.experiment.project + "/" + self.experiment.id
        run = api.run(key)
        
        for i in range(5):  # Retry up to 5 times with exponential backoff
            try:
                for artifact in run.logged_artifacts():
                    if 'best' in artifact.aliases:
                        # every x seconds and y epochs, tag the artifact with the best performance so far, such that it is not deleted.
                        if self.keep_every_epochs is not None:
                            current_epoch = self.experiment.summary.get("epoch")
                            if current_epoch is None:
                                print('ups epoch')
                            elif current_epoch - self.tagged_last_at_epochs >= self.keep_every_epochs-1:
                                tag_epochs = f"best_after_epoch_{current_epoch:05d}"
                                artifact.aliases.append(tag_epochs)
                                artifact.save()
                                self.tagged_last_at_epochs = current_epoch
                        
                        if self.keep_every_seconds is not None:
                            runtime = self.experiment.summary.get("_runtime")
                            if runtime is None:
                                print('ups runtime')
                            elif runtime - self.tagged_last_at_seconds >= self.keep_every_seconds:
                                runtime = int(runtime)
                                tag_seconds = f"best_after_sec_{runtime:08d}s__{format_seconds(runtime)}"
                                artifact.aliases.append(tag_seconds)
                                artifact.save()
                                self.tagged_last_at_seconds = runtime

                    if not artifact.aliases:
                        artifact.delete()
                        continue
                break
            except HTTPError as e:
                print(f"wandb artifact fetch failed (attempt {i+1}): {e}")
                time.sleep(3 ** i)
                
                
def format_seconds(seconds):
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{days:03d}d_{hours:02d}h_{minutes:02d}m_{seconds:02d}s"