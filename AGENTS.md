# Repository guidelines

This is a Python 3.12+ research codebase for iterative numerical solvers. Read the [root README](README.md) for the overview and the [1D](1D/README.md) or [2D](2D/README.md) guide before changing a workspace.

## Structure and conventions

- `1D/` contains its own driver, JSON configs, linear/nonlinear methods, diagnostics, collectors, and results. `2D/` contains a separate driver, declarative configuration and initialization, problem definitions, methods, diagnostics, tests, and results. Run their scripts from the corresponding workspace directory so local imports and paths resolve.
- `reference/` holds selected presentation, figure, report, and wavefront material. Numerical study records live under the relevant workspace's `results/` directory.
- Use 4-space indentation, `snake_case`, and explicit imports. Keep method names aligned with configuration values such as `p_unsplit`, `h_unsplit`, and `multigrid`. Put experiment settings in configs or clearly named study defaults, rather than solver loops.

## Setup and checks

- Use `uv sync` or create a virtual environment and run `python -m pip install -r requirements.txt`. Keep `requirements.txt` and `pyproject.toml` dependencies aligned.
- Run the deterministic 2D suite from `2D/`: `python -m unittest discover -s tests/unit -t . -p 'test_*.py'`. For solver changes, also run `python tests/smoke.py` or a bounded case through `main.py`.
- Run 1D configuration checks from `1D/` with `python load_config1D.py`; use a small configured run or focused collector/check for changes to 1D methods. Study collectors can be expensive and are not unit tests.

## Data and publication hygiene

- Treat `1D/results/thesis_report_data/` and the selected 2D thesis studies described in `2D/results/README.md` as frozen records. Write new runs to generated or explicitly chosen output paths; do not overwrite preserved data.
- Keep virtual environments, caches, temporary outputs, credentials, and machine-specific paths out of committed files. Commit generated results only when intentionally selected as an experiment record. Check public-facing material before staging.
- Use concise, imperative commit messages. For changes to numerical results, report the problem, method, config, verification command, and relevant output in the pull request.
