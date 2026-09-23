# Dynamic dual-energy CT grouped-count reconstruction: code and synthetic data

This repository contains the numerical code and synthetic data supporting the manuscript **A Statistical Analysis and Proximal Poisson Extension of Iterative Dynamic Dual-Energy CT** by Huahai Sun and Liang Li. It is a reproducibility record for the manuscript's scalar diagnostics and two-dimensional dynamic-group experiments. It contains no patient scans or measurements from a physical detector.

## Contents

| Path | Contents |
| --- | --- |
| `paper/code/dde_2d.py` | Four-bin grouped-Poisson simulation and reconstruction for SIRT, LS, weighted LS, and proximal Poisson updates. |
| `paper/code/summarize_2d.py` | Checks saved observations and reconstructions, recomputes summary tables, and draws the figures. |
| `paper/code/verify_counterexample.py` and `verify_local_relation.py` | Scalar LS/Poisson mismatch and local quadratic checks. |
| `paper/experiments/inputs_*.npz` | Grouped observations `y`, their expected means, and common initial images for two incident count levels and five fixed seeds. |
| `paper/experiments/recon_*.npz` | Final images from all four reconstruction methods for every count level and seed. |
| `paper/experiments/projector.npz` and `geometry_truth.npz` | Sparse projection operator, dynamic group partition, modified Shepp–Logan phantom, and true image channels. |
| `paper/experiments/trajectories.csv`, `results.json`, `runs.json`, `summary.*`, `verification.json`, and `protocol.json` | Per-iteration diagnostics, final metrics, experiment settings, and verification results. |
| `paper/data/*.csv` | Scalar diagnostic source data. |
| `paper/figures/*.pdf` and `*.png` | Figures regenerated from the saved data. |
| `lean/` | Lean 4 companion proofs for the allocation, majorization/descent, scalar Poisson, and mismatch claims. |
| `SHA256SUMS` | Checksums for the deposited code and numerical files. |

The experiment uses a 32 × 32 image grid, four fine energy bins, 60 views, 48 detector offsets, two count levels (1,000 and 20,000 incident photons per ray), and seeds 20260910–20260914. Each ray has two energy groups whose dividing threshold changes by view. The recorded arrays comprise 40 reconstructions and 4,000 outer updates. See `paper/experiments/protocol.json` for the complete parameter and environment record.

`geometry_truth.npz` stores `Xtrue` as a 1024 × 4 image matrix and `threshold` as one group boundary per ray. Each `inputs_*.npz` stores a 2588 × 2 array `y` of grouped counts and a corresponding `means` array. Each `recon_*.npz` contains four 1024 × 4 arrays named `SIRT`, `LS`, `WLS`, and `Poisson`. The coefficients are dimensionless. The data-generation and reconstruction grids are identical; the detector model assumes independent Poisson counts and excludes pile-up, charge sharing, and other real-detector effects.

## Reproduce the checks

Install [uv](https://docs.astral.sh/uv/) and run from this repository's root:

```sh
uv sync --locked
uv run python paper/code/verify_counterexample.py
uv run python paper/code/verify_local_relation.py
uv run python paper/code/summarize_2d.py
```

The last command verifies the archived two-dimensional arrays and trajectories, then regenerates `paper/experiments/summary.*`, `verification.json`, `table_2d.tex`, and figures. It does **not** rerun the 40 optimizations. To rerun them, use `uv run python paper/code/dde_2d.py` first; this overwrites the archived experiment files. The `--out` option sends a rerun to another directory if the archived outputs should be preserved. The environment is pinned in `pyproject.toml`, `.python-version`, and `uv.lock`.

The timed comparisons in `results.json` came from the recorded macOS arm64 run. Elapsed times depend on hardware and background load. The numerical experiment supports likelihood and convergence diagnostics; it is not clinical or detector-system validation.

## Lean companion

The `lean/` directory contains the source files, toolchain pin, and Lake configuration for Lean 4.33.1 with mathlib v4.33.1. To build it, follow [`lean/README.md`](lean/README.md) or run:

```sh
cd lean
lake update
lake exe cache get
lake build
```

The machine-checked statements cover grouped-count allocation, abstract majorization and descent, finite telescoping bounds, a scalar Poisson minimizer, and a scalar LS/Poisson mismatch counterexample. They do **not** formalize the manuscript's full nonsmooth subdifferential and Kurdyka–Łojasiewicz convergence proof.

## Availability and version

The files here are the synthetic source data and code for the manuscript version prepared in September 2026. Cite the repository URL together with the published article when available. A tagged version is intended to identify the exact submitted data snapshot. The manuscript's theoretical results are not fully machine-checked by this repository.
