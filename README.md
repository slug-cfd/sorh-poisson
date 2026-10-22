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
  configs/                # default and example run configurations
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

Change simulation parameters by editing `1D/configs/config_sample1D.json`, or pass another JSON file with:

```bash
cd 1D
uv run python main.py --config path/to/config.json
```

### Completed thesis data and new report-style runs

The final data used by the completed thesis are preserved under
[`1D/results/thesis_report_data/`](1D/results/thesis_report_data/). The report
plotter uses that frozen dataset by default, but writes recreated figures to
`1D/results/generated_report_figures/`.

To rerun Figures 1–4 with the collector that produced the frozen thesis data:

```bash
cd 1D
uv run python tests/collect_report_figure_data.py --figures 1 2 3 4
```

Each run gets a new directory under `1D/results/generated_report_data/comprehensive/`.
For smaller, individual runs, collect Figure 3 first and reuse its omega sweep:

```bash
cd 1D
uv run python tests/collect_figure3_data.py
uv run python tests/collect_figure1_data.py --omega-data results/generated_report_data/figure3_omega_sweeps.csv
uv run python tests/collect_figure2_data.py --omega-data results/generated_report_data/figure3_omega_sweeps.csv
uv run python tests/collect_figure4_data.py --omega-data results/generated_report_data/figure3_omega_sweeps.csv
```

The individual collectors write to `1D/results/generated_report_data/` and
overwrite their default CSVs on rerun; use `--output` to keep separate runs.
Use explicit data-directory options with `plot_report_figures.py` when plotting
newly collected data.

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
  configs/             # default sample, Poisson, and nonlinear examples
  methods/             # numerical methods by equation type
  problems/            # problem definitions and registries
  tests/               # smoke checks, validations, benchmarks, and studies
    unit/              # deterministic unittest suite
  diagnostics/         # plotting and analysis scripts
  results/             # frozen thesis data and ignored new output
```

Run the small default case or a named example:

```bash
cd 2D
uv run python main.py
uv run python main.py --config configs/config_poisson.json
uv run python main.py --config configs/config_nonlinear.json
```

The driver reads `configs/config_sample.json` by default and writes each run
to a new JSON file under `results/generated_runs/`. Use `--output` to choose a
path and `--overwrite` to replace an existing file deliberately. Plotting is
off in the sample config; set
`output.plot` or `output.plot_exact` in a config, or use
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

The frozen [two-blob and nonlinear study data](2D/results/README.md) support
the 2D thesis results. The [two-blob](2D/tests/two_blob_studies/README.md) and
[nonlinear](2D/tests/nonlinear_studies/README.md) study guides explain how to
collect new runs. The default driver and plotters write to ignored output
folders, leaving the frozen data unchanged.

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

The repository includes the frozen [1D thesis data](1D/results/thesis_report_data/)
and a [representative 1D solver record](1D/results/reference/sample_linear_sorh_run.json).
The [2D thesis data](2D/results/README.md) include the two-blob Poisson and
nonlinear diffusion studies, plus square and rectangular omega-formula checks.
New runs, working diagnostics, and archived results remain local by default.
