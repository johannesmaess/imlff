"""
train_deq_modular.py — Modular DEQ / explicit force-field training script.

Supports SchNet, PaiNN, SO3net representations.

Usage
-----
    python train_deq_modular.py [array_idx] [options]

Key CLI flags
~~~~~~~~~~~~~
    array_idx          SLURM job-array index; selects one config from the sweep
    --c INT            config_set preset (0-5) (you may add more)
    --rep STR          representation: SchNet | PaiNN | SO3net
    --use_deq BOOL     true / false
    --molecule STR     override molecule; sets molecule_names = [STR]
    --n_z INT          feature dimension
    --i INT            forw_max_iter (single value; wraps in a list)
    --cutoff FLOAT     message passing cutoff
    --force FLOAT      forces_in_loss weight [0, 1]
    --tied BOOL        shared_interactions
    --dev BOOL|'fast'  False logs to default wandb project. True to dev project. 'fast' has only very lightweight epochs.
    --resume ID        resume a wandb run (same run ID)
    --continue ID      start a new wandb run from a checkpoint
    --load ID          load config only (no checkpoint)

Loss overrides — applied after config_set and WandB; always win; work for fresh runs too
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    --fpc     at,strength,gamma   e.g.  --fpc 0,10000,0.4
    --trunc   at,strength,gamma   e.g.  --trunc 0,3.2,1.0
    --jac_reg at,strength         e.g.  --jac_reg 0,0.32
    'at' can be an integer epoch number or the literal 'true', which resolves
    to the epoch stored in the loaded checkpoint's wandb summary.

Python interface
~~~~~~~~~~~~~~~~
    cfg = parse_args_and_build_config()   # start from CLI defaults
    cfg.reps            = ['SO3net', 'PaiNN']  # sweep over two reps
    cfg.n_zs            = [128, 256]              # sweep over two sizes
    cfg.forw_max_iters  = [4, 6, 8]
    cfg.fpc_strengths   = [1e4, 1e5]
    training_configs = build_training_configs(cfg)  # Cartesian product
"""

from __future__ import annotations

import argparse
import copy
import datetime
import gc
import getpass
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from itertools import product as iterproduct
from typing import Any, Optional

import torch
import torch.nn.functional as F
import wandb

from schnetpack.datasets import MD17, MD22
from schnetpack.representation import PaiNN, SO3net, SchNet

from imlff.common.util import proj_path
from imlff.data.cumulene import Cumulene
from imlff.data.info import molecule_to_dataset
from imlff.data.util import load_md17_data
from imlff.loading import (
    artifact_or_run_cached,
    download_ckpt_from_wandb,
    load_model_from_wandb,
    turn_iter_into_deq,
)
from imlff.model.norms import (
    EquiUnitLengthFeaturesNorm,
    InvUnitLengthFeaturesNorm,
    EquiMergedLayerNorm,
)
from imlff.model.platform import wait_for_cuda
from imlff.model.util import init_iterating_schnet
from imlff.training.logger import WandBLoggerClearArtifacts
from imlff.training.run import train
from imlff.training.util import setup_task
from imlff.version import VERSION
from imlff.data.info import molecule_to_atomic_numbers


# ── representation classes ───────────────────────────────────────────────

REP_CLASSES: dict[str, type] = {
    "SchNet": SchNet,
    "PaiNN": PaiNN,
    "SO3net": SO3net,
}


# ── Module-level data ─────────────────────────────────────────────────────────

BATCH_SIZE_DICT: dict[str, dict[str, int]] = {
    "SO3net": {
        "Ac-Ala3-NHMe": 3, "DHA": 5, "AT-AT": 3, "stachyose": 2,
        "AT-AT-CG-CG": 3, "buckyball-catcher": 2, "double-walled_nanotube": 1,
        "aspirin": 10, "azobenzene": 5, "benzene": 10, "ethanol": 10,
        "malonaldehyde": 10, "naphthalene": 10, "paracetamol": 10,
        "salicylic_acid": 10, "toluene": 10, "uracil": 10,
    },
    "PaiNN": {
        "Ac-Ala3-NHMe": 5, "DHA": 5, "AT-AT": 5, "stachyose": 5,
        "AT-AT-CG-CG": 5, "buckyball-catcher": 3, "double-walled_nanotube": 1,
    },
    "SchNet": {"Ac-Ala3-NHMe": 10, "DHA": 8},
}

# Per-representation architectural defaults.
# Extend this dict when adding new representations.
# Fields that are sweep dimensions (cutoff, n_rbf, …) are applied by the caller
# before the sweep.  Fields that can vary per rep within a single sweep iteration
# (forw_max_iters, merged_norm, …) are applied inside the build_training_configs loop
# via rep_defs = REP_DEFAULTS.get(rep, {}); add new per-rep overrides there.
REP_DEFAULTS: dict[str, dict] = {
    "PaiNN": dict(
        cutoff=5.0, n_rbf=50, norm_for_deq=True, norm_for_explicit=False,
    ),
    "SO3net": dict(
        cutoff=5.0, n_rbf=50, norm_for_deq=True, norm_for_explicit=False,
        keep_every_epochs=50,
    ),
    "SchNet": dict(
        cutoff=5.0, n_rbf=50, norm_for_deq=True, norm_for_explicit=False,
    ),
}

_ACTIVATION_MAP: dict[str, Any] = {
    "silu": F.silu, "tanh": F.tanh, "sigmoid": F.sigmoid,
}


# ── ExperimentConfig ──────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """
    Single container for all experiment parameters.

    All sweep dimensions are lists; itertools.product runs over them.
    CLI flags set them to singletons. WandB config loading also sets singletons.
    The Python interface lets you assign arbitrary lists for sweeps.
    """

    # ── bookkeeping ────────────────────────────────────────────────────────────
    array_idx: int = 0
    global_seed: int = 0
    just_print: bool = False
    dev: Any = False                   # False | True | 'fast'
    config_set: int = 2

    # ── run continuation ───────────────────────────────────────────────────────
    resume: bool = False
    resume_id: Optional[str] = None
    continue_from_model_tag: Optional[str] = None
    load_config_from_model_tag: Optional[str] = None
    turn_into_deq: bool = False

    # ── data ───────────────────────────────────────────────────────────────────
    molecule_name: Optional[str] = None            # CLI arg, singular
    molecule_names: list = field(default_factory=lambda: list(molecule_to_atomic_numbers.keys()))
    num_train: Optional[int] = None
    num_val: Optional[int] = None
    batch_size: Optional[int] = None              # CLI override
    split_id: int = 0
    forces_in_loss: float = 0.95
    cumulene_dft: bool = False
    ls: Optional[tuple] = None                    # cumulene multi-dataset lengths

    # ── training loop ─────────────────────────────────────────────────────────
    max_epochs: int = 10000
    check_val_every_n_epoch: int = 5
    early_stopping: bool = True
    automatic_optimization: bool = True
    tags: list = field(default_factory=list)

    # ── architecture sweeps (lists → itertools.product) ────────────────────────
    reps: list = field(default_factory=lambda: ["PaiNN"])
    n_zs: list = field(default_factory=lambda: [128])
    cutoffs: list = field(default_factory=lambda: [5.0])
    n_rbfs: list = field(default_factory=lambda: [50])
    activations: list = field(default_factory=lambda: ["silu"])   # string keys
    implicit_diffs: list = field(default_factory=lambda: [False])
    scales: list = field(default_factory=lambda: [1.0])

    # ── iterator / shared-layer sweeps ────────────────────────────────────────
    use_deq: bool = True
    forw_max_iters: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8])
    back_max_iters: list = field(default_factory=lambda: [None])
    forw_tols: list = field(default_factory=lambda: [-1])
    back_tols: list = field(default_factory=lambda: [-1])
    shared_layers: list = field(default_factory=lambda: [False])
    painn_dont_share_filters_even_when_sharing_interactions: bool = False
    input_injection: bool = False

    # ── derived iterator lists (set by build_iterator_lists) ───────────────────
    n_interactions_list: list = field(default_factory=lambda: [1])
    shared_interactions_list: list = field(default_factory=lambda: [None])
    n_schnets_list: list = field(default_factory=lambda: [None])
    shared_schnets_list: list = field(default_factory=lambda: [None])

    # ── norm settings ──────────────────────────────────────────────────────────
    norm_for_deq: Any = True
    norm_for_explicit: Any = False
    norm_s: Any = False
    norm_c: Any = False
    norm_o: Any = False                            # set by build_norm_lists
    InvNorms: list = field(default_factory=lambda: [None])
    # EquiNorms are computed per-rep inside build_training_configs
    merged_norm: Optional[bool] = None  # if True: Inv/Equi disabled, EquiMergedLayerNorm used. Not logged to WandB.

    # ── regularization sweeps ─────────────────────────────────────────────────
    jac_reg: int = -1
    jac_reg_strengths: list = field(default_factory=lambda: [None])
    fpc: int = -1
    fpc_strengths: list = field(default_factory=lambda: [None])
    fpc_ats: list = field(default_factory=lambda: [None])
    fpc_gammas: list = field(default_factory=lambda: [1])
    trunc: int = -1
    trunc_strengths: list = field(default_factory=lambda: [None])
    trunc_ats: list = field(default_factory=lambda: [None])
    trunc_gammas: list = field(default_factory=lambda: [1])

    # ── CLI loss overrides (applied AFTER WandB loading) ──────────────────────
    # Tuples of (at, strength, gamma); 'at' can be int or True (→ checkpoint epoch).
    _fpc_cli: Optional[tuple] = None
    _trunc_cli: Optional[tuple] = None
    _jac_reg_cli: Optional[tuple] = None

    # ── internal flags ─────────────────────────────────────────────────────────
    # Prevent apply_config_set from overwriting explicit CLI values.
    _forw_max_iters_explicit: bool = False

    # EquiNorm class restored from WandB (used in build_training_configs).
    _wandb_EquiNorm: Optional[type] = None

    # ── WandB logging ──────────────────────────────────────────────────────────
    log_model: str = "all"
    clean_at_n_epoch: int = 50
    keep_every_seconds: int = 60 * 60 * 6
    keep_every_epochs: Optional[int] = None


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_loss_arg(s: str) -> tuple:
    """Parse a comma-separated loss-override argument.

    Format:  'at,strength[,gamma]'
    'at' is an int epoch or the literal 'true' (→ checkpoint epoch).
    Examples:
        '0,10000,0.4'     → fpc starts at epoch 0, strength 10000, gamma 0.4
        'true,100000,0.4' → fpc starts at the checkpoint's current epoch
        '0,0.32'          → jac_reg at epoch 0, strength 0.32 (no gamma)
    """
    parts = [p.strip() for p in s.split(",")]
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            f"Expected 'at,strength[,gamma]', got: {s!r}"
        )
    at_raw = parts[0]
    at = True if at_raw.lower() == "true" else int(at_raw)
    strength = float(parts[1])
    gamma = float(parts[2]) if len(parts) == 3 else None
    return (at, strength, gamma)


def parse_args_and_build_config() -> ExperimentConfig:
    """Parse CLI args and return a fully initialised ExperimentConfig."""
    cfg = ExperimentConfig(
        global_seed=torch.randint(0, 2**30, (1,)).item(),
        tags=[],
    )

    parser = argparse.ArgumentParser(description="Train DEQ / explicit force field")
    parser.add_argument("array_idx", type=int, nargs="?", default=0)
    parser.add_argument("--c", dest="config_set", type=int)
    parser.add_argument("--use_deq", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--rep", dest="rep", type=str)
    parser.add_argument("--molecule", dest="molecule_name", type=str, default=None)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--continue", dest="continue_id", type=str)
    parser.add_argument("--load", dest="load_config_from_model_tag", type=str)
    parser.add_argument("--n_z", type=int)
    parser.add_argument("--i", dest="forw_max_iter", type=int)
    parser.add_argument("--cutoff", type=float)
    parser.add_argument("--force", type=float, default=0.95)
    parser.add_argument("--num_train", type=int)
    parser.add_argument("--num_val", type=int)
    parser.add_argument(
        "--ls",
        type=lambda s: tuple(map(int, s.strip("()").split(","))),
        default=None,
        help="Cumulene chain lengths for multi-dataset, e.g. --ls '(1,2,3,4)'",
    )
    parser.add_argument("--tied", type=lambda x: x.lower() == "true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--dev",
        type=lambda x: True if x.lower() == "true" else (False if x.lower() == "false" else x),
        default=None,
    )
    # Loss overrides — applied after config_set and WandB; always win
    parser.add_argument(
        "--fpc", type=_parse_loss_arg, default=None, metavar="at,strength,gamma",
        help="FPC override: --fpc 0,10000,0.4  or  --fpc true,10000,0.4",
    )
    parser.add_argument(
        "--trunc", type=_parse_loss_arg, default=None, metavar="at,strength,gamma",
    )
    parser.add_argument(
        "--jac_reg", type=_parse_loss_arg, default=None, metavar="at,strength",
        help="Jacobian-reg override (no gamma): --jac_reg 0,0.32",
    )
    parser.add_argument(
        "--merged_norm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use EquiMergedLayerNorm instead of separate Inv/Equi norms.",
    )

    args = parser.parse_args()

    cfg.array_idx = args.array_idx
    print(f"Array index: {cfg.array_idx}")

    if args.use_deq is not None:
        cfg.use_deq = args.use_deq
        print(f"KWARG --- use_deq: {args.use_deq}")
    if args.rep:
        cfg.reps = [args.rep]
        print(f"KWARG --- rep: {args.rep}")
    if args.config_set is not None:
        cfg.config_set = args.config_set
        print(f"KWARG --- config_set: {args.config_set}")
    if args.molecule_name:
        cfg.molecule_name = args.molecule_name
        print(f"KWARG --- molecule: {args.molecule_name}")
    if args.resume:
        cfg.resume = True
        cfg.resume_id = args.resume
        print(f"KWARG --- resume: {args.resume}")
    if args.continue_id:
        cfg.resume = False
        cfg.continue_from_model_tag = args.continue_id
        print(f"KWARG --- continue: {args.continue_id}")
    if args.load_config_from_model_tag:
        cfg.load_config_from_model_tag = args.load_config_from_model_tag
        print(f"KWARG --- load config: {args.load_config_from_model_tag}")
    if args.n_z:
        cfg.n_zs = [args.n_z]
        print(f"KWARG --- n_z: {args.n_z}")
    if args.forw_max_iter is not None:
        cfg.forw_max_iters = [args.forw_max_iter]
        cfg._forw_max_iters_explicit = True
        print(f"KWARG --- forw_max_iters: {[args.forw_max_iter]}")
    if args.cutoff:
        cfg.cutoffs = [args.cutoff]
        print(f"KWARG --- cutoff: {args.cutoff}")
    cfg.forces_in_loss = args.force
    if args.force != 0.95:
        assert 0 <= args.force <= 1.0, "forces_in_loss must be in [0, 1]"
        print(f"KWARG --- forces_in_loss: {args.force}")
    if args.num_train:
        cfg.num_train = args.num_train
        print(f"KWARG --- num_train: {args.num_train}")
    if args.num_val:
        cfg.num_val = args.num_val
        print(f"KWARG --- num_val: {args.num_val}")
    if args.ls:
        cfg.ls = args.ls
        print(f"KWARG --- ls: {args.ls}")
    if args.tied is not None:
        cfg.shared_layers = [args.tied]
        print(f"KWARG --- tied: {args.tied}")
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
        print(f"KWARG --- batch_size: {args.batch_size}")
    if args.dev is not None:
        cfg.dev = args.dev
        print(f"KWARG --- dev: {args.dev}")
    if args.merged_norm is not None:
        cfg.merged_norm = args.merged_norm
        print(f"KWARG --- merged_norm: {args.merged_norm}")

    # Store raw loss overrides; resolved and applied after WandB loading.
    cfg._fpc_cli     = args.fpc
    cfg._trunc_cli   = args.trunc
    cfg._jac_reg_cli = args.jac_reg
    if args.fpc:
        print(f"KWARG --- fpc override (post-WandB): {args.fpc}")
    if args.trunc:
        print(f"KWARG --- trunc override (post-WandB): {args.trunc}")
    if args.jac_reg:
        print(f"KWARG --- jac_reg override (post-WandB): {args.jac_reg}")

    return cfg


# ── Tag / run-ID resolution ───────────────────────────────────────────────────

def resolve_run_tags(cfg: ExperimentConfig) -> None:
    """Expand short 8-char run IDs to full artifact tags. Enforce mutual exclusion."""
    if cfg.resume_id is not None and cfg.continue_from_model_tag is None:
        cfg.continue_from_model_tag = cfg.resume_id
    if cfg.resume_id is not None and len(cfg.resume_id) > 8:
        cfg.resume_id = cfg.resume_id.split(":v")[0].split("model-")[-1]

    assert not (cfg.load_config_from_model_tag and cfg.continue_from_model_tag), (
        "Cannot use --load and --continue simultaneously"
    )

    if cfg.continue_from_model_tag:
        cfg.load_config_from_model_tag = (
            cfg.continue_from_model_tag.split(":v")[0].split("model-")[-1]
        )
        if len(cfg.continue_from_model_tag) == 8:
            # Bare 8-char run id (e.g. "3yox0o6h"): resolve to the artifact
            # version with the highest pickled training epoch — NOT `:latest`,
            # which wandb auto-aliases to the most-recently-uploaded version
            # and can drift to a low-epoch ckpt logged mid-resume.
            #
            # pick_highest_epoch_artifact also backfills metadata['epoch'] on
            # any legacy artifacts it scans (one-time torch.load per artifact),
            # so subsequent resume picks for this run are cheap and correct.
            run_id = cfg.continue_from_model_tag
            run_tag = artifact_or_run_cached(run_id, run=True, return_tag=True)
            from imlff.loading.epoch_meta import pick_highest_epoch_artifact
            try:
                art, epoch = pick_highest_epoch_artifact(run_tag, backfill=True)
            except Exception as e:
                print(f"  ! pick_highest_epoch_artifact failed: {e}; falling back to :latest")
                art, epoch = None, None
            if art is not None:
                cfg.continue_from_model_tag = "/".join(
                    [art.entity, art.project, art.source_name]
                )
                print(
                    f"Continuing from: {cfg.continue_from_model_tag} "
                    f"(picked by highest epoch={epoch})"
                )
            else:
                try:
                    cfg.continue_from_model_tag = artifact_or_run_cached(
                        f"model-{run_id}:latest", return_tag=True
                    )
                    print("Continuing from (fallback :latest):", cfg.continue_from_model_tag)
                except Exception as e:
                    # No resolvable model artifact for this run. Start the model
                    # from scratch but keep resume_id / load_config_from_model_tag
                    # so wandb resumes (resume="must") and logs to the SAME run,
                    # using that run's stored config. Never crash here on resume —
                    # a run that died before logging any checkpoint is expected.
                    print(f"No logged model for {run_id!r} ({e}); starting fresh "
                          f"(will resume the same wandb run).")
                    cfg.continue_from_model_tag = None

    if cfg.load_config_from_model_tag:
        if len(cfg.load_config_from_model_tag) == 8:
            cfg.load_config_from_model_tag = artifact_or_run_cached(
                cfg.load_config_from_model_tag, run=True, return_tag=True
            )
        print("Loading config from:", cfg.load_config_from_model_tag)


# ── Config-set presets ────────────────────────────────────────────────────────

def apply_deq_defaults(cfg: ExperimentConfig) -> None:
    cfg.implicit_diffs = [False]
     # no regularisation:
    cfg.jac_reg = cfg.fpc = cfg.trunc = -1
    # but still set their default subparameters:
    cfg.fpc_ats = [True]
    cfg.trunc_ats = [[1, 2]]
    cfg.fpc_gammas = cfg.trunc_gammas = [1]
    cfg.jac_reg_strengths = [0.32]
    cfg.back_max_iters = [None]

    if cfg.use_deq and not cfg._forw_max_iters_explicit:
        # Default: aim for 10 iters at standard cutoffs; 20 for short cutoffs.
        # Rep-specific overrides are applied per-iteration inside build_training_configs.
        primary_cutoff = cfg.cutoffs[0]
        cfg.forw_max_iters = [10 if primary_cutoff > 2 else 20]
    
    
def apply_config_set(cfg: ExperimentConfig) -> None:
    """Apply experiment presets for cfg.config_set.

    This function is the primary place to define new sweeps.
    It only runs when NOT loading from a WandB config.
    """

    if cfg.config_set in [2, 3, 4, 5]:
        # 2 is the default jac_reg=0.32
        # 3 adds fpc
        # 4,5 are for cumulene training

        if cfg.use_deq:
            # apply jac_reg
            # Deliberate config: jac_reg + fpc
            cfg.jac_reg = 0
            cfg.trunc = cfg.fpc = -1
            cfg.jac_reg_strengths = [0.32]
            
            if cfg.config_set == 3:
                cfg.fpc = 0
                cfg.fpc_strengths = [10_000]
                cfg.fpc_gammas = [0.4]

        if cfg.config_set in [4, 5]:
            # cumulene training (5 trains on multiple)
            cfg.implicit_diffs = [False]
            cfg.tags += ["cumulene"]
            cfg.max_epochs = 10000
            cumulene_lengths = [1, 8, 2, 7, 3, 6, 4, 5]
            cfg.molecule_names = [f"cumulene{l}_train" for l in cumulene_lengths]
            print("Running cumulene:", cfg.molecule_names)

            if cfg.config_set == 5:
                cfg.tags += ["multi"]
                lengths = cfg.ls or cumulene_lengths
                cfg.molecule_names = [
                    f"cumulene{l}{'_dft' * cfg.cumulene_dft}_train" for l in lengths
                ]
                if cfg.dev == "fast":
                    cfg.molecule_names = cfg.molecule_names[:2]
                cfg.molecule_names = [sorted(cfg.molecule_names)]

            cfg.jac_reg = cfg.fpc = cfg.trunc = -1

        if cfg.config_set == 6:
            # deq norm development
            cfg.use_deq = True
            cfg.norm_for_deq = "equi"
            cfg.fpc = 0
            cfg.fpc_gammas = [1]
            cfg.fpc_ats = [[10]]
            cfg.fpc_strengths = [10_000]


    elif 100 <= cfg.config_set <= 200:
        cfg.molecule_names = ["aspirin"]
        cfg.tags += ["param search"] 
        cfg.check_val_every_n_epoch = 10
        cfg.max_epochs = 50000
        cfg.implicit_diffs = [False]

        # defaults
        cfg.jac_reg = cfg.fpc = cfg.trunc = -1
        cfg.jac_reg_strengths = cfg.fpc_strengths = cfg.trunc_strengths = [None] # non are running
        cfg.fpc_gammas = cfg.trunc_gammas = [1.] # default: no exponential decay
        cfg.fpc_ats = [True] # on all
        cfg.trunc_ats = [[1, 2]] # on first two (all are very expensive)
        
        if cfg.config_set == 100:
            pass # (vanilla)
        
        if cfg.config_set == 110:
            cfg.jac_reg = 0
            cfg.jac_reg_strengths = [0.01, 0.032, 0.1, 0.32, 1., 3.2, 10., 32., 100.]
        
        
        elif cfg.config_set in [120, 121]:
            cfg.fpc = 0
            cfg.fpc_strengths = [
                1_000,
                10_000,
                100_000,
                1_000_000,
                10_000_000,
                100_000_000
                ]
            
            if cfg.config_set == 121:
                cfg.fpc_gammas = [0.4]
        
        
        elif cfg.config_set in [130, 131]:
            cfg.trunc = 0
            cfg.trunc_strengths = [0.01, .1, 1., 10., 100.]
            
            if cfg.config_set == 131:
                cfg.trunc_gammas = [0.4]
        
        

# ── Derived list builders ─────────────────────────────────────────────────────

def build_molecule_names(cfg: ExperimentConfig) -> None:
    """Override molecule_names with CLI --molecule singleton if given."""
    if cfg.molecule_name:
        cfg.molecule_names = [cfg.molecule_name]


def build_iterator_lists(cfg: ExperimentConfig) -> None:
    """Derive n_interactions_list, n_schnets_list etc. from use_deq + input_injection."""
    if not cfg.use_deq:
        if not cfg.input_injection:
            cfg.n_interactions_list = cfg.forw_max_iters
            cfg.n_schnets_list = [1]
            cfg.shared_schnets_list = [True]
            cfg.shared_interactions_list = (
                cfg.shared_layers
                if isinstance(cfg.shared_layers, list)
                else [cfg.shared_layers]
            )
        else:
            cfg.tags += ["ii"]
            cfg.n_interactions_list = [1]
            cfg.n_schnets_list = cfg.forw_max_iters
            cfg.shared_schnets_list = (
                cfg.shared_layers
                if isinstance(cfg.shared_layers, list)
                else [cfg.shared_layers]
            )
            cfg.shared_interactions_list = [False]
        cfg.forw_max_iters = [None]
        assert cfg.back_max_iters == [None]
    else:
        cfg.n_interactions_list = [1]
        cfg.n_schnets_list = [None]
        cfg.shared_schnets_list = [None]
        cfg.shared_interactions_list = [None]


def build_norm_lists(cfg: ExperimentConfig) -> None:
    """Derive norm_o and InvNorms from use_deq and norm flags.

    EquiNorms are computed per-rep inside build_training_configs because
    SchNet does not support equivariant norms.
    """
    cfg.norm_o = cfg.norm_for_deq if cfg.use_deq else cfg.norm_for_explicit
    cfg.InvNorms = [None]
    if (
        cfg.norm_s is True or cfg.norm_s == "inv"
        or cfg.norm_c is True or cfg.norm_c == "inv"
        or cfg.norm_o is True or cfg.norm_o == "inv"
    ):
        cfg.InvNorms = [
            InvUnitLengthFeaturesNorm
        ]


# ── WandB config loading ──────────────────────────────────────────────────────

def maybe_load_wandb_config(cfg: ExperimentConfig) -> tuple[Optional[dict], Any]:
    """Load config from a WandB run and override all relevant cfg fields.

    Returns (continue_from_config, continue_from_run).
    Both are None if no WandB tag is set.
    continue_from_run is the wandb Run object; its .summary['epoch'] is used
    by apply_cli_loss_overrides when '--fpc true,...' is given.
    """
    if not cfg.load_config_from_model_tag:
        cfg.split_id = 0
        return None, None

    continue_from_run = wandb.Api().run(cfg.load_config_from_model_tag)
    wc = continue_from_run.config

    # ── data ──────────────────────────────────────────────────────────────────
    cfg.molecule_names = [wc["molecule"]]
    cfg.split_id = wc["split_id"]
    cfg.forces_in_loss = wc["forces_in_loss"]
    cfg.num_train = wc["num_train"]
    cfg.num_val = wc["num_val"]
    cfg.batch_size = cfg.batch_size or wc["batch_size"]

    # ── architecture ──────────────────────────────────────────────────────────
    cfg.n_zs = [wc["n_z"]]
    if cfg.turn_into_deq:
        assert cfg.use_deq and not wc["use_deq"]
    else:
        cfg.use_deq = wc["use_deq"]

    cfg.n_schnets_list = [wc["n_schnets"]]
    cfg.shared_schnets_list = [wc["shared_schnets"]]
    cfg.n_interactions_list = [wc["n_interactions"]]
    cfg.shared_interactions_list = [wc["shared_interactions"]]

    rep_name = wc["AtomisticRepresentationClass"]
    cfg.reps = [rep_name]
    
    cfg.activations = [wc.get("activation", "silu")]
    cfg.cutoffs = [wc["cutoff"]]
    cfg.n_rbfs = [wc["n_rbf"]]

    # ── norms ─────────────────────────────────────────────────────────────────
    cfg.norm_s = wc.get("norm_s", False)
    cfg.norm_c = wc.get("norm_c", False)
    cfg.norm_o = wc.get("norm_o", True)  # old default was True

    _norm_cls = {
        "InvUnitLengthFeaturesNorm":   InvUnitLengthFeaturesNorm,
        "EquiUnitLengthFeaturesNorm":  EquiUnitLengthFeaturesNorm,
    }
    cfg.InvNorms = [_norm_cls.get(wc.get("InvNormClass"), None)]
    cfg._wandb_EquiNorm = _norm_cls.get(wc.get("EquiNormClass"), None)
    # Restore merged-norm choice from the original run (not CLI-overridable when loading).
    cfg.merged_norm = wc.get("MergedNormClass") not in (None, "None", "")

    # ── DEQ params ────────────────────────────────────────────────────────────
    cfg.implicit_diffs = [wc.get("implicit_diff", False)]
    cfg.forw_tols      = [wc.get("forw_tol", None)]
    cfg.back_tols      = [wc.get("back_tol", None)]
    cfg.forw_max_iters = [wc.get("forw_max_iter", None)]
    cfg.back_max_iters = [wc.get("back_max_iter", None)]

    # ── regularization ────────────────────────────────────────────────────────
    cfg.jac_reg            = wc.get("jac_reg", -1)
    cfg.jac_reg_strengths  = [wc.get("jac_reg_strength", None)]
    cfg.fpc                = wc.get("fpc", -1)
    cfg.trunc              = wc.get("trunc", -1)
    cfg.fpc_strengths      = [wc.get("fpc_strength", None)]
    cfg.trunc_strengths    = [wc.get("trunc_strengths", None)]
    cfg.fpc_ats            = [wc.get("fpc_at", None)]
    cfg.trunc_ats          = [wc.get("trunc_at", None)]
    fpcg  = wc.get("fpc_gamma");  cfg.fpc_gammas  = [fpcg  if fpcg  is not None else 1]
    truncg = wc.get("trunc_gamma"); cfg.trunc_gammas = [truncg if truncg is not None else 1]

    cfg.tags = list(dict.fromkeys(list(continue_from_run.tags) + cfg.tags))
    cfg.tags.append("resume" if cfg.resume else "continue")

    return wc, continue_from_run


# ── CLI loss overrides ────────────────────────────────────────────────────────

def apply_cli_loss_overrides(cfg: ExperimentConfig, continue_from_run: Any) -> None:
    """Apply --fpc / --trunc / --jac_reg CLI overrides.

    Works for both fresh runs and --continue/--load runs.
    These args are applied after config_set and WandB loading so they always win.
    'at=true' requires continue_from_run (needs checkpoint epoch); integer 'at' works
    without any checkpoint.
    """
    def _resolve_at(at: Any) -> int:
        if at is True:
            assert continue_from_run is not None, (
                "'true' in a loss arg requires --continue or --load (needs checkpoint epoch)"
            )
            epoch = continue_from_run.summary.get("epoch")
            assert epoch is not None, "Checkpoint has no 'epoch' in wandb summary"
            return int(epoch)
        return at

    if cfg._fpc_cli is not None:
        at, strength, gamma = cfg._fpc_cli
        gamma_eff = gamma if gamma is not None else 1.0
        resolved_at = _resolve_at(at)
        # If --fpc true,S,G matches the strength/gamma already in the loaded wandb
        # config, keep the previously scheduled fpc epoch when it's still in the
        # future (ckpt_epoch < wandb_fpc); otherwise use the ckpt epoch.
        if at is True and continue_from_run is not None:
            wc = continue_from_run.config
            wandb_strength = wc.get("fpc_strength")
            wandb_gamma_raw = wc.get("fpc_gamma")
            wandb_gamma = wandb_gamma_raw if wandb_gamma_raw is not None else 1.0
            wandb_fpc_epoch = wc.get("fpc", -1)
            if (
                wandb_strength == strength
                and wandb_gamma == gamma_eff
                and wandb_fpc_epoch is not None
                and wandb_fpc_epoch >= 0
            ):
                ckpt_epoch = resolved_at
                resolved_at = max(ckpt_epoch, int(wandb_fpc_epoch))
                print(
                    f"CLI --fpc matches loaded run (strength={strength}, gamma={gamma_eff}); "
                    f"ckpt_epoch={ckpt_epoch}, wandb fpc={wandb_fpc_epoch} → fpc={resolved_at}"
                )
        cfg.fpc            = resolved_at
        cfg.fpc_strengths  = [strength]
        cfg.fpc_gammas     = [gamma_eff]
        cfg.fpc_ats        = [True]
        cfg.tags += ["fpc late in training"]
        print(f"CLI override → fpc={cfg.fpc}, strength={strength}, gamma={gamma}")

    if cfg._trunc_cli is not None:
        at, strength, gamma = cfg._trunc_cli
        cfg.trunc           = _resolve_at(at)
        cfg.trunc_strengths = [strength]
        cfg.trunc_gammas    = [gamma if gamma is not None else 1.0]
        cfg.trunc_ats       = [True]
        cfg.tags += ["trunc late in training"]
        print(f"CLI override → trunc={cfg.trunc}, strength={strength}, gamma={gamma}")

    if cfg._jac_reg_cli is not None:
        at, strength, _gamma = cfg._jac_reg_cli
        cfg.jac_reg           = _resolve_at(at)
        cfg.jac_reg_strengths = [strength]
        print(f"CLI override → jac_reg={cfg.jac_reg}, strength={strength}")


# ── Tag cleanup and reg sanitization ─────────────────────────────────────────

def cleanup_tags_and_reg(cfg: ExperimentConfig) -> None:
    """Sanitize regularization sweep lists, append descriptive tags.

    Always run last — after all reg params are finalised (config_set or WandB).
    Disabled components are zeroed out so the product loop generates fewer configs.
    """
    # Jacobian regularization
    active_jac = (
        cfg.jac_reg >= 0
        and cfg.jac_reg_strengths != [None]
        and any(s is not None and s > 0 for s in cfg.jac_reg_strengths)
    )
    if not active_jac:
        cfg.jac_reg = -1
        cfg.jac_reg_strengths = [None]

    # Fixed-point correction
    active_fpc = (
        cfg.fpc >= 0
        and any(s or 0 > 0 for s in cfg.fpc_strengths)
        and any(a for a in cfg.fpc_ats if a)
    )
    if not active_fpc:
        cfg.fpc = -1
        cfg.fpc_strengths = cfg.fpc_ats = [None]

    # Truncation
    active_trunc = (
        cfg.trunc >= 0
        and any(s or 0 > 0 for s in cfg.trunc_strengths)
        and any(g for g in cfg.trunc_gammas if g)
    )
    if not active_trunc:
        cfg.trunc = -1
        cfg.trunc_strengths = cfg.trunc_ats = [None]

    # Tolerance tags
    if cfg.forw_tols == [0]:     cfg.tags += ["0 tol"]



# ── Data config helper ────────────────────────────────────────────────────────

def _build_data_config(
    molecule_name: Any,
    layer_calls: int,
    *,
    num_train: Optional[int],
    num_val: Optional[int],
    dev: Any,
    batch_size: Optional[int],
    rep: str,
    split_id: int,
    cutoff: float,
) -> dict:
    """Build dataset config dict. All formerly-closed-over variables are explicit."""
    if isinstance(molecule_name, list):
        for m in molecule_name:
            assert "cumulene" in m, f"Expected cumulene molecule, got {m!r}"
        Dataset = Cumulene
    else:
        Dataset = molecule_to_dataset(molecule_name, name=False)

    if Dataset is MD17:
        curr_num_train = num_train or 950
        curr_num_val   = num_val   or 50
    elif Dataset is MD22:
        big = molecule_name in ("buckyball-catcher", "double-walled_nanotube")
        curr_num_train = num_train or (3000 if big else 5000)
        curr_num_val   = num_val   or 1000
    elif Dataset is Cumulene:
        curr_num_train = num_train or (10 if dev == "fast" else 4500)
        curr_num_val   = num_val   or (10 if dev == "fast" else 450)
    else:
        raise ValueError(f"Unknown dataset for molecule {molecule_name!r}")

    if batch_size is not None:
        curr_batch_size = batch_size
    elif isinstance(molecule_name, list):
        curr_batch_size = max(1, int(25 / len(molecule_name)))
        print(f"batch_size={curr_batch_size} for cumulene multi-dataset ({len(molecule_name)} mols)")
    else:
        curr_batch_size = BATCH_SIZE_DICT.get(rep, {}).get(
            molecule_name, 10 if rep == "SO3net" else 25
        )
        if layer_calls < 10:
            curr_batch_size = int(curr_batch_size * (10 / layer_calls - 1)) or 1
        if molecule_name == "double-walled_nanotube" and rep == "SO3net" and curr_batch_size > 5:
            curr_batch_size = 5

    return dict(
        Dataset=Dataset,
        num_train=curr_num_train if dev != "fast" else 10,
        num_val=curr_num_val   if dev != "fast" else 10,
        molecule=molecule_name,
        batch_size=curr_batch_size,
        split_id=split_id,
        adj_cutoff=torch.inf if Dataset is MD17 else cutoff,
        num_workers=int(16 / (1 if isinstance(molecule_name, str) else len(molecule_name))),
    )


# ── Training config generation ────────────────────────────────────────────────

def build_training_configs(cfg: ExperimentConfig) -> list[tuple]:
    """Generate the full Cartesian product of (molecule, model_config, …) tuples."""
    # Determine whether equivariant norms are needed and which class to use.
    # SchNet does not use EquiNorm; handled per-rep in the loop.
    use_equi_norm = (
        cfg.norm_s is True or cfg.norm_s == "equi"
        or cfg.norm_c is True or cfg.norm_c == "equi"
        or cfg.norm_o is True or cfg.norm_o == "equi"
    )
    _equi_norm_cls = (
        EquiUnitLengthFeaturesNorm
    ) if use_equi_norm else None
    # WandB-loaded config may specify a different class; honour it.
    if cfg._wandb_EquiNorm is not None:
        _equi_norm_cls = cfg._wandb_EquiNorm

    training_configs: list[tuple] = []

    for (
        rep, n_z, cutoff, n_rbf, activation_str, implicit_diff,
        n_schnets, shared_schnets, n_interactions, shared_interactions,
        InvNorm,
        forw_max_iter, back_max_iter,
        forw_tol, back_tol,
        jac_reg_strength, trunc_strength, fpc_strength,
        fpc_at, trunc_at, fpc_gamma, trunc_gamma,
        scale, molecule_name,
    ) in iterproduct(
        cfg.reps,
        cfg.n_zs,
        cfg.cutoffs,
        cfg.n_rbfs,
        cfg.activations,
        cfg.implicit_diffs,
        cfg.n_schnets_list,
        cfg.shared_schnets_list,
        cfg.n_interactions_list,
        cfg.shared_interactions_list,
        cfg.InvNorms,
        cfg.forw_max_iters,
        cfg.back_max_iters,
        cfg.forw_tols,
        cfg.back_tols,
        cfg.jac_reg_strengths,
        cfg.trunc_strengths,
        cfg.fpc_strengths,
        cfg.fpc_ats,
        cfg.trunc_ats,
        cfg.fpc_gammas,
        cfg.trunc_gammas,
        cfg.scales,
        cfg.molecule_names,
    ):
        # ── Apply per-rep defaults from REP_DEFAULTS ──────────────────────────
        rep_defs = REP_DEFAULTS.get(rep, {})
        if cfg.use_deq and not cfg._forw_max_iters_explicit:
            forw_max_iter = rep_defs.get("forw_max_iters", [forw_max_iter])[0]

        # ── Validity filters ───────────────────────────────────────────────────
        if cfg.use_deq and n_schnets is not None:              continue
        if not shared_schnets and n_schnets == 1:              continue
        if shared_interactions and n_interactions == 1:        continue
        if cfg.use_deq and forw_max_iter == 1:                 continue

        # implicit_diff requires DEQ with n_interactions=1 and forw_max_iter∈{9,10}
        if implicit_diff:
            if not cfg.use_deq or n_interactions != 1:        continue
            max_iters_limit = (
                9 if (
                    isinstance(molecule_name, str)
                    and molecule_name == "double-walled_nanotube"
                    and rep == "SO3net"
                ) else 10
            )
            if forw_max_iter != max_iters_limit:
                print(f'Warning: Skipping config bc. of wrong iteration count: {forw_max_iter}!')
                continue

        # ── Rep-specific adjustments ───────────────────────────────────────────

        # shared_filters only applies to PaiNN
        if rep != "PaiNN":
            shared_filters = None
        else:
            shared_filters = (
                not cfg.painn_dont_share_filters_even_when_sharing_interactions
                if shared_interactions else False
            )

        # EquiNorm: not applicable for SchNet.
        EquiNorm = _equi_norm_cls if rep != "SchNet" else None

        # Merged equivariant layer norm: overrides Inv/Equi norms when enabled.
        merged_norm_eff = (
            cfg.merged_norm
            if cfg.merged_norm is not None
            else rep_defs.get("merged_norm", False)
        )
        if merged_norm_eff:
            MergedNorm = EquiMergedLayerNorm
            InvNorm_eff = None
            EquiNorm_eff = None
        else:
            MergedNorm = None
            InvNorm_eff = InvNorm
            EquiNorm_eff = EquiNorm

        # OOM guard for SO3net on double-walled nanotube
        actual_forw_max_iter = forw_max_iter
        if (
            cfg.use_deq
            and isinstance(molecule_name, str)
            and molecule_name == "double-walled_nanotube"
            and rep == "SO3net"
            and actual_forw_max_iter is not None
            and actual_forw_max_iter > 9
        ):
            print(f"Capping forw_max_iter {actual_forw_max_iter}→9 for SO3net double-walled_nanotube")
            actual_forw_max_iter = 9

        Ds_str = (
            "Cumulene" if isinstance(molecule_name, list)
            else molecule_to_dataset(molecule_name)
        )

        # ── model_config ───────────────────────────────────────────────────
        activation_fn = _ACTIVATION_MAP[activation_str]
        model_config: dict[str, Any] = dict(
            n_z=n_z,
            use_deq=cfg.use_deq,
            n_schnets=n_schnets,
            shared_schnets=shared_schnets,
            n_interactions=n_interactions,
            activation=activation_fn,
            shared_interactions=shared_interactions,
            shared_filters=shared_filters,
            InvNormClass=InvNorm_eff,
            EquiNormClass=EquiNorm_eff,
            MergedNormClass=MergedNorm,
            norm_s=cfg.norm_s,
            norm_c=cfg.norm_c,
            norm_o=cfg.norm_o,
            cutoff=cutoff,
            n_rbf=n_rbf,
            AtomisticRepresentationClass=REP_CLASSES[rep],
        )
        if cfg.use_deq:
            model_config.update(
                implicit_diff=implicit_diff,
                forw_max_iter=actual_forw_max_iter,
                back_max_iter=back_max_iter or actual_forw_max_iter,
                forw_tol=forw_tol,
                back_tol=back_tol,
            )

        layer_calls = n_interactions * (
            actual_forw_max_iter if cfg.use_deq else (n_schnets or 1)
        )

        # ── further_arguments ─────────────────────────────────────────────
        further_arguments = dict(
            scale=scale,
            layer_calls=layer_calls,
            layer_params=(1 if shared_interactions else n_interactions) * (
                1 if (cfg.use_deq or shared_schnets) else (n_schnets or 1)
            ),
            seed=cfg.global_seed + len(training_configs),
            slurm_job_id=int(os.environ.get("SLURM_JOB_ID", -1)),
            slurm_node_name=os.environ.get("SLURMD_NODENAME", "unknown"),
        )

        # ── reg_kwarg ─────────────────────────────────────────────────────
        train_md_steps, val_md_steps = -1, 2
        if Ds_str == "MD22" and rep in ("SO3net", "PaiNN"):
            val_md_steps = -1
        if not cfg.automatic_optimization:
            train_md_steps = 0

        reg_kwarg = dict(
            jac_reg=cfg.jac_reg,
            jac_reg_strength=jac_reg_strength,
            fpc=cfg.fpc,
            trunc=cfg.trunc,
            fpc_strength=fpc_strength,
            trunc_strength=trunc_strength,
            fpc_at=fpc_at,
            trunc_at=trunc_at,
            fpc_gamma=fpc_gamma,
            trunc_gamma=trunc_gamma,
            md_steps=(train_md_steps, val_md_steps),
        ) if cfg.use_deq else {}

        data_config = _build_data_config(
            molecule_name, layer_calls,
            num_train=cfg.num_train,
            num_val=cfg.num_val,
            dev=cfg.dev,
            batch_size=cfg.batch_size,
            rep=rep,
            split_id=cfg.split_id,
            cutoff=cutoff,
        )

        training_configs.append(
            (molecule_name, model_config, reg_kwarg, further_arguments, data_config)
        )

    return training_configs


# ── Config printing & selection ───────────────────────────────────────────────

def _print_and_select_config(cfg: ExperimentConfig, training_configs: list) -> int:
    """Print all configs (or just the selected one) and return array_idx."""
    print("=" * 50)
    print(f"Total configs: {len(training_configs)}")

    for i, (mol, mc, _, _, _) in enumerate(training_configs):
        if cfg.dev or i == cfg.array_idx or cfg.just_print:
            print("=" * 50)
            print(f"Config {i}:  molecule={mol}")
            for k, v in mc.items():
                print(f"  {k}: {v}")
            nnpot_dummy, *_ = init_iterating_schnet(**mc)
            n_params = sum(p.numel() for p in nnpot_dummy.parameters())
            print(f"  Num. parameters: {n_params}")

    print("=" * 50)
    if cfg.resume_id:
        assert len(training_configs) == 1, "--resume requires exactly one config"
        return 0
    if len(training_configs) == 1:
        print("Only one config, using it.")
        return 0
    print(f"Using config {cfg.array_idx}")
    return cfg.array_idx


# ── WandB setup ───────────────────────────────────────────────────────────────

def setup_wandb_and_logger(
    cfg: ExperimentConfig,
    model_config: dict,
    data_config: dict,
    reg_kwarg: dict,
    further_arguments: dict,
    run_name: str,
) -> tuple:
    """Initialise wandb run and logger, log all hyperparams + git hash."""
    wandb_project = "dev-v10" + ("-foo" if cfg.dev else "")
    run = wandb.init(
        **(dict(id=cfg.resume_id, resume="must") if cfg.resume else {}),
        entity="qcml-group",
        project=wandb_project,
        name=run_name,
        mode="online",
        tags=list(dict.fromkeys(cfg.tags)),  # dedup, preserve order
    )
    keep_every_epochs_eff = (
        cfg.keep_every_epochs
        if cfg.keep_every_epochs is not None
        else min(
            (REP_DEFAULTS.get(r, {}).get("keep_every_epochs", 250) for r in cfg.reps),
            default=250,
        )
    )
    logger = WandBLoggerClearArtifacts(
        log_model=cfg.log_model,
        keep_every_epochs=keep_every_epochs_eff,
        keep_every_seconds=cfg.keep_every_seconds,
    )
    logger._project_name = wandb_project
    logger._clean_at_n_epoch = cfg.clean_at_n_epoch

    logger.log_hyperparams({"model_key": run_name, "forces_in_loss": cfg.forces_in_loss})
    logger.log_hyperparams(model_config)
    logger.log_hyperparams(data_config)
    logger.log_hyperparams(reg_kwarg)
    logger.log_hyperparams(further_arguments)
    logger.log_hyperparams({f: getattr(VERSION, f) for f in VERSION._fields})

    try:
        git_dir = proj_path(".git")
        commit = subprocess.check_output(
            ["git", "--git-dir", git_dir, "rev-parse", "--short", "HEAD"]
        ).strip().decode()
        dirty = subprocess.check_output(
            ["git", "--git-dir", git_dir, "status", "--porcelain"]
        ).strip().decode()
        if dirty:
            commit += "-dirty"
        logger.log_hyperparams({"commit": commit})
    except Exception as e:
        print("Could not log git hash:", e)

    return run, logger


# ── Model loading / initialisation ───────────────────────────────────────────

def load_and_init_model(
    cfg: ExperimentConfig,
    model_config: dict,
    further_arguments: dict,
    device: str,
    run: Any,
    logger: Any,
    continue_from_config: Optional[dict],
) -> tuple:
    """Download checkpoint if continuing; otherwise initialise a fresh model.

    Returns (nnpot, ckpt_path). ckpt_path is None when starting fresh.
    """
    ckpt_path = None

    if cfg.continue_from_model_tag:
        try:
            ckpt_path = download_ckpt_from_wandb(cfg.continue_from_model_tag)
        except Exception as e:
            print("Error loading checkpoint:", e)
            print("Initialising fresh model instead.")
            assert not cfg.turn_into_deq

        if ckpt_path:
            nnpot = load_model_from_wandb(cfg.continue_from_model_tag, map_location=device)

            if cfg.turn_into_deq:
                turn_iter_into_deq(nnpot)

            if hasattr(nnpot.representation, "deq"):
                deq = nnpot.representation.deq
                deq.implicit_diff  = model_config["implicit_diff"]
                deq.forw_tol       = model_config["forw_tol"]
                deq.back_tol       = model_config["back_tol"]
                deq.forw_max_iter  = model_config["forw_max_iter"]
                deq.back_max_iter  = model_config["back_max_iter"]

            # Tag the source artifact with a back-reference to this new run.
            date_str = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
            art = artifact_or_run_cached(cfg.continue_from_model_tag)
            art.aliases += [f"continued-from--{run.id}--{date_str}"]
            art._update()

            cont_mt = cfg.continue_from_model_tag
            if "latest" in cont_mt or "best" in cont_mt:
                version_string = art._source_name.split(":")[-1]
                cont_mt = cont_mt.replace("latest", version_string).replace("best", version_string)
            logger.log_hyperparams({
                "continue_from_model_tag": cfg.continue_from_model_tag,
                "continue_from_config": continue_from_config,
                f"continue_from_model_tag--{date_str}": cont_mt,
            })

    if ckpt_path is None:
        nnpot, _, _ = init_iterating_schnet(device=device, **model_config)
        scale = further_arguments["scale"]
        if scale != 1:
            for param in nnpot.representation.f.parameters():
                param.data.mul_(scale)

    return nnpot, ckpt_path


# ── Train-kwargs assembly ─────────────────────────────────────────────────────

def _build_train_kwargs(
    cfg: ExperimentConfig,
    logger: Any,
    model_config: dict,
    data_config: dict,
    ckpt_path: Optional[str],
) -> dict:
    check_val = cfg.check_val_every_n_epoch
    rep_cls = model_config.get("AtomisticRepresentationClass")
    rep_name = rep_cls.__name__ if rep_cls is not None else ""
    if rep_name == "SO3net" and data_config.get("molecule") == "double-walled_nanotube":
        check_val = 1

    train_kwargs = dict(
        accelerator="cuda",       # overwritten by main() with actual device
        max_epochs=cfg.max_epochs,
        logger=logger,
        log_every_n_steps=10,
        check_val_every_n_epoch=check_val if not cfg.dev else 1,
        num_sanity_val_steps=2,
        ckpt_path=ckpt_path,
        early_stopping=cfg.early_stopping,
    )
    if cfg.dev == "fast":
        train_kwargs.update(
            check_val_every_n_epoch=1,
            limit_train_batches=10,
            limit_val_batches=20,
            max_epochs=1000 if ckpt_path else 6,
            log_every_n_steps=1,
        )
    return train_kwargs


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Starting train_deq_modular.py...")
    torch.set_float32_matmul_precision("medium")

    # 1. CLI → ExperimentConfig
    cfg = parse_args_and_build_config()

    # 2. Expand run-ID tags
    resolve_run_tags(cfg)

    # 3. Config-set presets (only if NOT loading from WandB)
    if not cfg.load_config_from_model_tag:
        apply_deq_defaults(cfg)
        apply_config_set(cfg)
        build_molecule_names(cfg)
        build_iterator_lists(cfg)
        build_norm_lists(cfg)

    # 4. Load WandB config (overrides almost everything if set)
    continue_from_config, continue_from_run = maybe_load_wandb_config(cfg)

    # 5. CLI loss overrides — override config_set AND WandB, always applied last
    #    (--fpc true,... requires a checkpoint; integer at works for fresh runs too)
    apply_cli_loss_overrides(cfg, continue_from_run)

    # 6. Sanitize reg params and append tags
    cleanup_tags_and_reg(cfg)

    # 7. Build Cartesian product of all sweep configs
    training_configs = build_training_configs(cfg)

    array_idx = _print_and_select_config(cfg, training_configs)
    if cfg.just_print:
        return

    molecule_name, model_config, reg_kwarg, further_arguments, data_config = (
        training_configs[array_idx]
    )

    torch.manual_seed(further_arguments["seed"])
    device = wait_for_cuda(timeout=2 if cfg.dev else 30)
    print(f"Using device: {device}")

    local_prefix = f"{getpass.getuser()}_" if device != "cuda" else ""
    model_key = init_iterating_schnet(**model_config, get_key=True)

    run, logger = setup_wandb_and_logger(
        cfg, model_config, data_config, reg_kwarg, further_arguments,
        local_prefix + model_key,
    )

    nnpot, ckpt_path = load_and_init_model(
        cfg, model_config, further_arguments, device, run, logger, continue_from_config,
    )

    train_kwargs = _build_train_kwargs(cfg, logger, model_config, data_config, ckpt_path)
    train_kwargs["accelerator"] = device

    # Reload before deepcopy: turn_into_deq must be applied at the start of fit.
    if ckpt_path:
        nnpot = load_model_from_wandb(cfg.continue_from_model_tag, map_location=device)
    nnpot = copy.deepcopy(nnpot)

    # ── Training loop with full error recovery ────────────────────────────────
    last_epoch = -1
    current_hour = None
    hit_count = 0
    retry_ckpt_path = f"/tmp/retry_{run.id}.ckpt"
    train_kwargs["retry_ckpt_path"] = retry_ckpt_path
    molecule_data = load_md17_data(**data_config, duplicate_for_run=True)
    task = None

    while True:
        try:
            if torch.cuda.is_available():
                print(f"GPU alloc before attempt: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            task = setup_task(
                nnpot,
                forces_in_loss=cfg.forces_in_loss,
                turn_into_deq=cfg.turn_into_deq,
                automatic_optimization=cfg.automatic_optimization,
                check_val_every_n_epoch=train_kwargs.get("check_val_every_n_epoch") or 1,
                **reg_kwarg,
            )
            task.log("val_loss", torch.nan)   # prevent EarlyStopping from firing on step 0
            print("Training...")
            last_epoch = train(task=task, datamodule=molecule_data, **train_kwargs)
            break

        except KeyboardInterrupt:
            print("Keyboard interrupt — stopping.")
            break

        except Exception as e:
            err_repr = f"{type(e).__name__}: {e}"
            is_oom = (
                isinstance(e, torch.OutOfMemoryError)
                or "out of memory" in str(e).lower()
                or "deepcopy protocol" in str(e).lower()
            )

            # Resume from the checkpoint train() saved on failure (preserves epoch,
            # optimizer, scheduler). Drop refs to task + exception so their GPU
            # tensors and traceback frames are freed before the next attempt.
            if os.path.exists(retry_ckpt_path):
                train_kwargs["ckpt_path"] = retry_ckpt_path
            try:
                traceback.clear_frames(e.__traceback__)
            except Exception:
                pass
            del task, e
            task = None
            # Clear DEQ state that holds refs to the previous forward graph
            # (see deq.py: self._last_inputs is set during forward and only
            # released after the capture_g_bar backward hook fires — so a
            # backward OOM leaves a full forward graph attached to nnpot).
            for m in nnpot.modules():
                if hasattr(m, "_last_inputs"):
                    del m._last_inputs
            for p in nnpot.parameters():
                p.grad = None
            # Round-trip through CPU to drop any remaining GPU refs attached
            # to nnpot via hooks/closures. Params reload from ckpt on retry.
            nnpot = nnpot.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            nnpot = nnpot.to(device)
            if torch.cuda.is_available():
                print(f"GPU alloc after cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB")

            if is_oom:
                bs = data_config["batch_size"]
                if bs == 1:
                    print("OOM at batch_size=1; cannot reduce further.")
                    break
                new_bs = bs // 2
                data_config["batch_size"] = new_bs
                molecule_data.batch_size = new_bs
                molecule_data.val_batch_size = new_bs
                molecule_data._train_dataloader = None
                molecule_data._val_dataloader = None
                logger.log_hyperparams({"batch_size": new_bs})
                print(f"OOM: reducing batch_size {bs} → {new_bs}")
                continue

            print("Exception during training:")
            traceback.print_exc()
            if current_hour != (now_hour := datetime.datetime.now().hour):
                current_hour = now_hour
                hit_count = 0
            hit_count += 1
            if hit_count > 10:
                print("Too many errors this hour — stopping.")
                break
            sleep_time = min(60, 2 ** hit_count)
            print(f"\nError #{hit_count} this hour ({err_repr}). Retrying in {sleep_time}s...")
            time.sleep(sleep_time)

    print(f"Ran {last_epoch + 1} of {train_kwargs['max_epochs']} epochs.")
    if "script_complete" not in run.tags:
        run.tags = (*run.tags, "script_complete")


if __name__ == "__main__":
    main()
