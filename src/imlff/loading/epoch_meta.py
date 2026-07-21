"""Authoritative epoch lookup for wandb model artifacts.

Background — why `:latest` and `:vNN` cannot be trusted:
  * wandb auto-aliases every newly logged artifact with `:latest`, even when a
    resumed run logs a near-zero-epoch checkpoint that should NOT supersede the
    high-epoch ckpt the previous run finished on.
  * The `:vNN` version number is monotonic by upload order, not by training
    epoch. A resumed run can upload v77 at epoch 50 *after* v76 at epoch 299.
  * The `epoch{NNNNN}` tag on artifacts is written by logger.py at tag-write
    time using the run's CURRENT summary epoch, so old artifacts get re-tagged
    by later resumed runs and lie about what's inside.

The only ground truth is what's pickled inside `model.ckpt` itself
(`ckpt['epoch']`, `ckpt['global_step']`). This module exposes:

  * `read_ckpt_epoch_step(path)`   — torch.load a local ckpt and return (epoch, step).
  * `stamp_artifact_epoch(artifact, epoch, global_step)` — write to metadata + save.
  * `get_artifact_epoch(artifact, backfill=True)` — read metadata, backfilling
    by downloading the ckpt if absent. Returns None on failure.
  * `pick_highest_epoch_artifact(run, ...)` — scan a run's logged model
    artifacts and return the one with the highest authoritative epoch.

Backfill is intentionally side-effecting: once we've paid the cost of figuring
out the true epoch for a legacy artifact, we write it to metadata so future
callers (run_val, train_deq_modular resume) get the answer for free.

Note: the filename pattern `epoch=N-step=M.ckpt` is deliberately NOT used as a
fallback. Lightning fills it from a template at save time, and we've observed
artifacts where the filename disagrees with the pickled `ckpt['epoch']`. The
only trusted sources are `artifact.metadata['epoch']` (written by this module
at train time) and `torch.load(...)['epoch']` (the pickled ground truth).
"""

from __future__ import annotations

from typing import Optional

import torch
import wandb


def read_ckpt_epoch_step(ckpt_path: str) -> tuple[Optional[int], Optional[int]]:
    """torch.load the ckpt and return (epoch, global_step). (None, None) on failure."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  ! read_ckpt_epoch_step failed for {ckpt_path}: {e}")
        return None, None
    return ckpt.get("epoch"), ckpt.get("global_step")


def stamp_artifact_epoch(artifact, epoch: int, global_step: Optional[int]) -> None:
    """Write epoch / global_step into artifact.metadata and persist.

    No-op if metadata already has matching values.
    """
    md = artifact.metadata or {}
    cur_epoch = md.get("epoch")
    cur_step = md.get("global_step")
    if cur_epoch == epoch and cur_step == global_step:
        return
    md["epoch"] = int(epoch)
    if global_step is not None:
        md["global_step"] = int(global_step)
    artifact.metadata = md
    try:
        artifact.save()
    except Exception as e:
        print(f"  ! stamp_artifact_epoch save failed for {artifact.name}: {e}")


def get_artifact_epoch(artifact, backfill: bool = True) -> Optional[int]:
    """Return the authoritative epoch for an artifact.

    Resolution order:
      1. artifact.metadata['epoch'] if present.
      2. (if backfill) download the ckpt, torch.load, read ckpt['epoch'],
         stamp into metadata, return.

    Returns None if all attempts fail.
    """
    md = artifact.metadata or {}
    if "epoch" in md and md["epoch"] is not None:
        return int(md["epoch"])

    if not backfill:
        return None

    # Download + torch.load. Import here to avoid a circular import at module
    # load time (loading/__init__.py imports a lot).
    from imlff.loading import download_ckpt_from_wandb

    tag = "/".join([artifact.entity, artifact.project, artifact.source_name])
    try:
        ckpt_path = download_ckpt_from_wandb(tag, verbose=False)
    except Exception as e:
        print(f"  ! get_artifact_epoch: download failed for {tag}: {e}")
        return None
    epoch, step = read_ckpt_epoch_step(ckpt_path)
    if epoch is None:
        return None
    stamp_artifact_epoch(artifact, epoch, step)
    return int(epoch)


def pick_highest_epoch_artifact(
    run_tag: str,
    *,
    backfill: bool = True,
    only_model: bool = True,
):
    """Scan a wandb run's logged artifacts and return the one with max epoch.

    Backfills metadata['epoch'] for any artifact that's missing it (legacy
    artifacts) so subsequent calls are cheap.

    Returns (artifact, epoch) or (None, None) if no model artifacts found.
    """
    api = wandb.Api()
    run = api.run(run_tag)
    best_art = None
    best_epoch = -1
    for art in run.logged_artifacts():
        if only_model and "model" not in art.source_name:
            continue
        epoch = get_artifact_epoch(art, backfill=backfill)
        if epoch is None:
            continue
        if epoch > best_epoch:
            best_epoch = epoch
            best_art = art
    if best_art is None:
        return None, None
    return best_art, best_epoch
