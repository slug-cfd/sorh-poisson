# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.12 research codebase for numerical solver experiments. The `2D/` directory contains the two-dimensional code: `2D/main.py` is the entry point, `2D/configuration/` contains declarative settings, JSON loading, runtime types, and explicit initialization, and the JSON files in `2D/configs/` contain reusable settings. `2D/problems/` stores problem definitions, and `2D/methods/` stores solver implementations including `2D/methods/poisson/`, `2D/methods/nonlinear_diffusion/`, and `2D/methods/linear_diffusion.py`. The `2D/tests/unit/` directory contains deterministic unit tests; the top level of `2D/tests/` contains smoke checks, validation scripts, and benchmarks, while named subdirectories contain larger studies. `2D/diagnostics/` contains analysis and plotting scripts. The `1D/` directory contains the one-dimensional code: `1D/main.py` is the entry point, `1D/config1D.py`, `1D/load_config1D.py`, and `1D/configs/config_sample1D.json` handle runtime configuration, `1D/methods/` stores solvers, `1D/tests/` contains test and benchmark helpers, and `1D/diagnostics/` contains analysis, plotting, and spectral-radius scripts. Generated outputs are written to `2D/results/` and `1D/results/`; avoid committing new large result files unless they are intentionally part of an experiment record. Older implementations are kept in `archive/`.

## Build, Test, and Development Commands

Install dependencies with:

```bash
uv sync
```

Run the default 2D simulation from `config_sample.json`:

```bash
cd 2D
uv run python main.py
```

Run config loading as a quick sanity check:

```bash
cd 2D
uv run python -c 'from configuration import load_config; print(load_config("config_sample.json"))'
```

Use `2D/tests/smoke.py` for a bounded solver check. Run the directional-ridge benchmark with:

```bash
cd 2D
uv run python tests/tests_directional_ridge.py
```

For ad hoc 1D runs, use:

```bash
cd 1D
uv run python main.py
```

Run 1D config loading as a quick sanity check:

```bash
cd 1D
uv run python load_config1D.py
```

The completed thesis data are frozen under `1D/results/thesis_report_data/`.
Use the report-style collectors under `1D/tests/` only for new experiments;
they write separately under `1D/results/generated_report_data/`:

```bash
cd 1D
uv run python tests/collect_figure3_data.py
uv run python diagnostics/plot_report_figures.py --help
uv run python diagnostics/plot_problem_setup.py --help
```

## Coding Style & Naming Conventions

Use 4-space indentation and keep code compatible with Python 3.12. Prefer explicit imports for project modules; avoid wildcard imports in new code unless matching an existing local pattern is necessary. Use `snake_case` for functions, variables, and module names. Keep solver names aligned with config values such as `p_unsplit`, `h_ADI`, and `multigrid` so JSON configs map cleanly to implementations. Store tunable experiment parameters in config objects or JSON rather than hard-coding them inside solver loops.

## Testing Guidelines

There is no configured coverage target. Run the deterministic 2D unit suite from `2D/` with `uv run python -m unittest discover -s tests/unit -t . -p 'test_*.py'`. Validate broader solver changes with `2D/tests/smoke.py` or a small `2D/config_sample.json` case through `2D/main.py`. Add isolated behavior checks under `2D/tests/unit/`; reserve the top-level test scripts for bounded smoke checks, validations, and benchmarks.

## Commit & Pull Request Guidelines

Recent commit messages are short, imperative summaries such as `added ADI implementations` and `refactoring + multigrid methods`. Follow that style: describe the numerical method, bug fix, or refactor in one concise line. Pull requests should include the problem/method tested, the command run, relevant config values, and any result files or plots needed to interpret the change.

## Security & Configuration Tips

Do not commit local virtual environments, IDE files, caches, or temporary outputs. Keep reusable experiment settings in `2D/config_sample.json`, `1D/configs/config_sample1D.json`, or a clearly named config file, and avoid embedding machine-specific paths in source files.
