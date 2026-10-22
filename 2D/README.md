# Two-dimensional experiments

This workspace tests SORh, SORp, ADI, and multigrid methods on Poisson and diffusion problems. It includes two-blob Poisson and nonlinear-diffusion studies used in the thesis, plus smaller validation and diagnostic scripts.

## Layout

| Path | Purpose |
| --- | --- |
| `main.py`, `runner.py` | Run a configured simulation and save its record. |
| `configuration/`, `configs/` | Declarative settings, JSON loading, initialization, and examples. |
| `problems/`, `methods/` | Problem definitions and numerical solvers. |
| `tests/unit/` | Deterministic unit tests. |
| `tests/` | Smoke checks, validations, benchmarks, and named studies. |
| `diagnostics/` | Analysis and plotting scripts. |
| `results/` | Selected thesis records and generated output. |

`load_config()` reads declarative settings; `initialize_simulation()` builds the solver-ready grid, problem, and arrays. Complete the [repository setup](../README.md#setup), then enter `2D/` from the repository root.

## Run and test

```bash
cd 2D
python main.py
python main.py --config configs/config_poisson.json
python main.py --config configs/config_nonlinear.json
python tests/smoke.py
python -m unittest discover -s tests/unit -t . -p 'test_*.py'
```

The default is `configs/config_sample.json`. Each run writes a new JSON record under ignored `results/generated_runs/`. Use `--output` to choose a path and `--overwrite` to replace an existing file. Plotting is disabled in the sample config; control it with `output.plot` and `output.plot_exact` in JSON, or `--plot`/`--plot-exact` and their `--no-` forms on the command line.

For nonlinear diffusion, `problem.coefficient` selects `regularized` or `constant`, and `problem.initial_condition` selects `zero` or `linear_x`. The default regularization value is `0.0`; positive values reduce degeneracy. `methods.p_unsplit.coefficient_mode` selects `pointwise` (default) or `frozen_per_sweep`. Poisson cases can override declared constants through `problem.parameters` for parameter studies.

## Results and reproduction

The [results guide](results/README.md) identifies the frozen two-blob and nonlinear thesis studies and the omega-formula checks. The [two-blob](tests/two_blob_studies/README.md) and [nonlinear](tests/nonlinear_studies/README.md) guides explain how to collect new data. The default driver and plotters write separately from the preserved records.

Run standalone diagnostics as modules from `2D/`; use `--help` for inputs and output locations:

```bash
python -m diagnostics.plot_two_blob_problem --help
python -m diagnostics.plot_two_blob_results --help
python -m diagnostics.plot_nonlinear_thesis_results --help
python -m diagnostics.diagnose_polar_star --help
python -m diagnostics.analyze_sorh_spectrum --help
```

These commands also work with `uv run python` in place of `python`.
