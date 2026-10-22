# SORh numerical solver research

This repository contains the Python code and selected research records behind a thesis on hyperbolic successive over-relaxation (SORh). It compares SORh with parabolic SOR (SORp) and other iterative methods on one- and two-dimensional problems. The included data and figures support inspection of the thesis results; new experiments can be run independently of those records.

The thesis itself, *SORh: Hyperbolic Relaxation Methods For Elliptic Problems In Computational Fluid Dynamics*, can be found [here](https://escholarship.org/uc/item/5gc906gb). 

## Repository map

| Directory | Contents |
| --- | --- |
| [1D/](1D/README.md) | Linear and nonlinear one-dimensional solvers, relaxation studies, diagnostics, and thesis figure data. |
| [2D/](2D/README.md) | Config-driven Poisson and diffusion solvers, multigrid comparisons, tests, and two-blob/nonlinear thesis studies. |
| [reference/](reference/README.md) | Thesis presentation, selected figures, technical reports, and wavefront material. |

Each solver workspace has its own configurations, methods, tests, diagnostics, and results. Its README explains how to run examples and locate the preserved thesis data.

## Setup

Use Python 3.12 or newer. Check that `python3 --version` meets this requirement, then run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, use `py -3.12 -m venv .venv` and activate with `.venv\Scripts\activate`. If you use `uv`, run `uv sync` from the repository root. Commands in the workspace guides use `python` with the virtual environment active; `uv run python` also works.

The repository currently has no reuse license. Contact the author for permission before reusing the code or research materials.
