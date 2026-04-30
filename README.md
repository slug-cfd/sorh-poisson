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

Run the 1D benchmark/test helper:

```bash
cd 1D
uv run python tests/tests_1D.py
```

The benchmark script supports options such as `--Ns`, `--iter-max`, `--omega-min`, `--omega-max`, and `--output-dir`. Generated result analysis lives in `1D/diagnostics/`; for example:

```bash
cd 1D
uv run python diagnostics/analysis.py --input-dir results/sor_performance
```

## 2D

The `2D/` directory contains the two-dimensional solver experiments.

```text
2D/
  main.py                 # main config-driven 2D driver
  config.py               # 2D config dataclasses and defaults
  load_config.py          # JSON config loader
  config_sample.json      # default run configuration
  config_storage.py       # stored method/omega tables
  run_record.py           # run result/record dataclasses
  BC_2D.py                # boundary condition helpers
  methods/                # Poisson and toy-problem method implementations
  problems/               # problem definitions and Poisson test registry
  tests/                  # test and benchmark scripts
  diagnostics/            # analysis, plotting, and diagnostics scripts
  results/                # generated 2D outputs
```

Run the default 2D simulation:

```bash
cd 2D
uv run python main.py
```

You can also run it from the repository root:

```bash
uv run python 2D/main.py
```

Check that the 2D config loads correctly:

```bash
cd 2D
uv run python load_config.py
```

Change simulation parameters by editing `2D/config_sample.json`. The main parameters are the selected `problem`, `poisson_problem`, `method`, grid size, `omega`, iteration limit, multigrid flag, and hyperbolic-direction controls. The main driver reads this JSON and writes a run record to `2D/results/test_record.json`.

Run the directional-ridge benchmark:

```bash
cd 2D
uv run python tests/tests_directional_ridge.py
```

General 2D helper routines live in `2D/tests/tests_2D.py`. Plotting and diagnostics scripts live in `2D/diagnostics/`.

## Outputs

Generated outputs are written under each workspace’s `results/` directory:

```text
1D/results/
2D/results/
```

These directories are for experiment records, benchmark CSVs, summaries, and plots. Avoid committing large generated outputs unless they are intentionally part of an experiment record.
