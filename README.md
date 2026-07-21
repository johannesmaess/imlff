# Implicit Machine Learning Force Fields (I-MLFF)

Train, test, and simulate molecular dynamics with I-MLFF.

## Installation

This repository integrates with SchNetPack, specifically a forked version with increased modularity.
To install both: 
```
git clone git@github.com:johannesmaess/imlff.git
cd imlff
git submodule init
git submodule update
pip install -e .
pip install -e ./schnetpack
```

## Training and evaluation of I-MLFF

Scripts reproducing 'Implicit Machine Learning Force Fields Accelerate Molecular Dynamics Simulations' are the following.

| Result | Generating file |
|---|---|
| Training Implicit/explicit models on MD17/MD22/Cumulene datasets | [`src/imlff/scripts/train.py`](src/imlff/scripts/train.py) |
| I-MLFF implementation with broad features, including support for training | [`src/imlff/model/deq.py`](src/imlff/model/deq.py) |
| I-MLFF inference, akin to SI Algorithm 1 | [`src/imlff/model/deq_inference.py`](src/imlff/model/deq_inference.py) |
| Polynomial extrapolation of fixed-points, including SI Algorithm 2 | [`src/imlff/md/stateful_extrapolator.py`](src/imlff/md/stateful_extrapolator.py) |
| Validation and Testing of models | [`src/imlff/scripts/fetch_runs.py`](src/imlff/scripts/fetch_runs.py), [`src/imlff/scripts/run_val.py`](src/imlff/scripts/run_val.py), [`src/imlff/scripts/run_test.py`](src/imlff/scripts/run_test.py) |
| Accelerated molecular dynamics | [`src/imlff/scripts/run_md.py`](src/imlff/scripts/run_md.py) |
| Fig. 1a — Fixed-point continuity | [`notebooks/analysis_dataset_and_fixpoint_continuity.ipynb`](notebooks/analysis_dataset_and_fixpoint_continuity.ipynb) |
| Fig. 1b — Fixed-point extrapolation | [`notebooks/plot_streamline.ipynb`](notebooks/plot_streamline.ipynb) |
| Fig. 1c — Acceleration summary (empirical) | [`notebooks/analysis_accuracy_readout.ipynb`](notebooks/analysis_accuracy_readout.ipynb) |
| Fig. 2a — Detailed accuracy analysis | [`notebooks/analysis_accuracy_readout.ipynb`](notebooks/analysis_accuracy_readout.ipynb) |
| Fig. 2b–d — Tolerance and iteration of I-MLFF under regularization & warmstarts | [`notebooks/analysis_warmstart.ipynb`](notebooks/analysis_warmstart.ipynb) |
| Fig. 3a — Accuracy using warmstarts, compared with explicit models | [`notebooks/analysis_warmstart.ipynb`](notebooks/analysis_warmstart.ipynb) |
| Fig. 3b,c — Drift & stability given solver tolerance | [`notebooks/analysis_md_energy_qualities.ipynb`](notebooks/analysis_md_energy_qualities.ipynb) |
| Fig. 4a–c — Adaptive iteration depth visualized on Ac-Ala3-NHMe | [`notebooks/analysis_md_ala3.ipynb`](notebooks/analysis_md_ala3.ipynb) |
| SI Fig. S1 — Long-range effects and generalization in cumulene molecules | [`notebooks/analysis_cumulene.ipynb`](notebooks/analysis_cumulene.ipynb) |
| SI Fig. S5 — Increased solver iterations at high potential energies & temperatures | [`notebooks/analysis_md_ala3.ipynb`](notebooks/analysis_md_ala3.ipynb) |
| SI Tables 2,3 — Force accuracy of models | [`notebooks/analysis_accuracy_readout.ipynb`](notebooks/analysis_accuracy_readout.ipynb) |
| SI Fig. S6 — Peak memory consumption of explicit and implicit models | [`notebooks/analysis_memory_consumption.ipynb`](notebooks/analysis_memory_consumption.ipynb) |
| SI Fig. S7 — Implicit model illustration in 1D: the magnetic Ising model | [`src/imlff/model/deq_inference.py`](src/imlff/model/deq_inference.py) |