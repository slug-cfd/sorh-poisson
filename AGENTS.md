# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.12 research codebase for numerical solver experiments. The `2D/` directory contains the two-dimensional code: `2D/main.py` is the entry point, `2D/config.py`, `2D/load_config.py`, and `2D/config_sample.json` handle runtime configuration, `2D/problems/` stores problem definitions, and `2D/methods/` stores solver implementations including `2D/methods/Poisson/` and `2D/methods/ToyProblem/`. The `2D/tests/` directory contains 2D test and benchmark helpers, while `2D/diagnostics/` contains analysis and plotting scripts. The `1D/` directory contains the one-dimensional code: `1D/main.py` is the entry point, `1D/config1D.py`, `1D/load_config1D.py`, and `1D/config_sample1D.json` handle runtime configuration, `1D/methods/` stores solvers, `1D/tests/` contains test and benchmark helpers, and `1D/diagnostics/` contains analysis, plotting, and spectral-radius scripts. Generated outputs are written to `2D/results/` and `1D/results/`; avoid committing new large result files unless they are intentionally part of an experiment record. Older implementations are kept in `archive/`.

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
uv run python load_config.py
```

Use `2D/tests/tests_2D.py` as a helper module for experiment sweeps and comparisons; it is not currently a pytest-style suite. Run the directional-ridge benchmark with:

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

Use `1D/tests/tests_1D.py` for 1D experiment generation and `1D/diagnostics/analysis.py` for generated-result analysis:

```bash
cd 1D
uv run python tests/tests_1D.py --Ns 16 32
uv run python diagnostics/analysis.py
```

## Coding Style & Naming Conventions

Use 4-space indentation and keep code compatible with Python 3.12. Prefer explicit imports for project modules; avoid wildcard imports in new code unless matching an existing local pattern is necessary. Use `snake_case` for functions, variables, and module names. Keep solver names aligned with config values such as `p_unsplit`, `h_ADI`, and `multigrid` so JSON configs map cleanly to implementations. Store tunable experiment parameters in config objects or JSON rather than hard-coding them inside solver loops.

## Testing Guidelines

There is no configured coverage target or formal test framework. Validate changes by running a small `2D/config_sample.json` case through `2D/main.py` and checking convergence output plus `2D/results/test_record.json`. When adding solver behavior, add a small deterministic helper or comparison in `2D/tests/` and keep runtime bounded by using modest grid sizes such as `Nx=32`, `Ny=32`.

## Commit & Pull Request Guidelines

Recent commit messages are short, imperative summaries such as `added ADI implementations` and `refactoring + multigrid methods`. Follow that style: describe the numerical method, bug fix, or refactor in one concise line. Pull requests should include the problem/method tested, the command run, relevant config values, and any result files or plots needed to interpret the change.

## Security & Configuration Tips

Do not commit local virtual environments, IDE files, caches, or temporary outputs. Keep reusable experiment settings in `2D/config_sample.json`, `1D/config_sample1D.json`, or a clearly named config file, and avoid embedding machine-specific paths in source files.
