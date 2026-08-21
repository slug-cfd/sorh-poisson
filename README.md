# Research Solver Experiments

This repository contains Python experiments for iterative numerical solvers in one and two spatial dimensions. The code is organized into separate `1D/` and `2D/` workspaces. Each workspace has its own main driver, configuration file, solver implementations, test helpers, diagnostics, and output directory.

Install dependencies from the repository root with:

```bash
uv sync
```

## 1D

The `1D/` directory contains the one-dimensional solver experiments.

```text
1D/
  main.py                 # main config-driven 1D driver
  config1D.py             # 1D config dataclasses and defaults
  load_config1D.py        # JSON config loader
  config_sample1D.json    # default run configuration
  solver1D.py             # shared 1D solver runner
  problem_helpers.py      # exact/reference helpers
  methods/                # linear and nonlinear method implementations
  tests/                  # test and benchmark generation scripts
  diagnostics/            # analysis, plotting, and spectral diagnostics
  results/                # generated 1D outputs
```

Run the default 1D simulation:

```bash
cd 1D
uv run python main.py
```

Check that the 1D config loads correctly:

```bash
cd 1D
uv run python load_config1D.py
```

Change simulation parameters by editing `1D/config_sample1D.json`, or pass another JSON file with:

```bash
cd 1D
uv run python main.py --config path/to/config.json
```

### Report figure data

The collectors for the 1D report data (Figures 3.1-3.4) live in `1D/tests/`. Run Figure 3 first to create
the shared omega-sweep dataset, then reuse that dataset for Figures 1, 2, and 4
to avoid repeating solver runs:

```bash
cd 1D
uv run python tests/collect_figure3_data.py
uv run python tests/collect_figure1_data.py --omega-data results/final_report_figure_data/consistency_rerun_20260801/figure3_omega_sweeps.csv
uv run python tests/collect_figure2_data.py --omega-data results/final_report_figure_data/consistency_rerun_20260801/figure3_omega_sweeps.csv
uv run python tests/collect_figure4_data.py --omega-data results/final_report_figure_data/consistency_rerun_20260801/figure3_omega_sweeps.csv
```

`omega_sweep.py` provides the shared sweep routine. By default, all four
collectors write concise CSV files under
`1D/results/final_report_figure_data/consistency_rerun_20260801/`.

Generate the report figures from the collected data, or create the standalone
problem-setup and spectral-radius figures, with:

```bash
uv run python diagnostics/plot_report_figures.py --help
uv run python diagnostics/plot_problem_setup.py --help
uv run python diagnostics/plot_spectral_radius_overlay.py --help
```

## 2D

The `2D/` workspace contains config-driven Poisson, nonlinear-diffusion, and
linear-diffusion solvers.

The configuration pipeline is explicit: `load_config()` returns declarative
`Config` settings, and `initialize_simulation()` builds the solver-ready
`SimulationContext` containing the grid, resolved problem, and arrays.

```text
2D/
  main.py              # command-line entry point
  configuration/       # settings, JSON loading, and explicit initialization
  config_sample.json   # small default Poisson run
  configs/             # named Poisson and nonlinear examples
  methods/             # numerical methods by equation type
  problems/            # problem definitions and registries
  tests/               # smoke checks, validations, benchmarks, and studies
    unit/              # deterministic unittest suite
  diagnostics/         # plotting and analysis scripts
  results/             # generated output (normally not committed)
```

Run the small default case or a named example:

```bash
cd 2D
uv run python main.py
uv run python main.py --config configs/config_poisson.json
uv run python main.py --config configs/config_nonlinear.json --output results/nonlinear_record.json
```

The driver writes `results/test_record.json` by default. Plotting is off by
default; set `output.plot` or `output.plot_exact` in a config, or use
`--plot`/`--plot-exact` on the command line. Command-line flags override the
config and have matching `--no-plot` forms.

For nonlinear diffusion, `problem.coefficient` selects `regularized` or
`constant` coefficients and `problem.initial_condition` selects `zero` or
`linear_x`. The regularization default `0.0` gives `u/(1+u)`; positive values
reduce degeneracy while preserving `v(1)=0.5`. For nonlinear `p_unsplit`, set
`methods.p_unsplit.coefficient_mode` to `pointwise` (default) or
`frozen_per_sweep`.

Poisson cases can override their declared constants through
`problem.parameters`; diagnostics use this for parameter studies without
modifying registered problem definitions.

Run the bounded smoke check or the unit suite:

```bash
uv run python tests/smoke.py
uv run python -m unittest discover -s tests/unit -t . -p 'test_*.py'
```

Run standalone 2D diagnostics as package modules from `2D/`; use `--help` to
see their inputs and output locations:

```bash
uv run python -m diagnostics.plot_two_blob_problem --help
uv run python -m diagnostics.plot_two_blob_results --help
uv run python -m diagnostics.plot_nonlinear_thesis_results --help
uv run python -m diagnostics.diagnose_polar_star --help
uv run python -m diagnostics.analyze_sorh_spectrum --help
```

## Outputs

Generated outputs are written under each workspace’s `results/` directory:

```text
1D/results/
2D/results/
```

These directories are for experiment records, benchmark CSVs, summaries, and plots. Avoid committing large generated outputs unless they are intentionally part of an experiment record.
