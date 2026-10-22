# Nonlinear SOR parameter studies

These are manual, preliminary studies for lexicographic `h_unsplit` and
`p_unsplit`. Run commands from `2D`. Data collection and plotting are kept
separate.

Commands below use `uv run python`; after the [repository setup](../../../README.md#setup), an activated virtual environment can use `python` in its place.

Run the omega calibration first:

```bash
uv run python -m tests.nonlinear_studies.collect omega
```

Run one or more targeted `N=64` SORp omega cases without repeating the full
calibration:

```bash
uv run python -m tests.nonlinear_studies.run_n64_omega --omega 2.50
uv run python -m tests.nonlinear_studies.run_n64_omega --omega 2.45 2.50 2.55
```

Later studies automatically use the newest completed omega result, or accept
an explicit result directory with `--omega-results`:

```bash
uv run python -m tests.nonlinear_studies.collect timestep
uv run python -m tests.nonlinear_studies.collect h-cfl
uv run python -m tests.nonlinear_studies.collect degeneracy
uv run python -m tests.nonlinear_studies.collect initial
uv run python -m tests.nonlinear_studies.collect performance
```

Reoptimize SORp specifically for the $u_0=x$ initial guess at $N=32$, while
retaining the original fixed-omega initial-guess study:

```bash
uv run python -m tests.nonlinear_studies.optimize_initial_guess_sorp
```

This performs a coarse-to-fine search only for the linear-in-$x$ guess and
writes a three-case comparison under
`results/nonlinear_studies/initial_sorp_optimized/`.

Each collection command creates a timestamped directory under
`results/nonlinear_studies/<study>/`. Studies with distinct search trials and
selected cases contain `all_runs.csv`; every study contains its named
comparison/selection CSV and `settings.json`.

After the degeneracy study, generate its derived table and plot separately:

```bash
uv run python -m tests.nonlinear_studies.plot_degeneracy \
  results/nonlinear_studies/degeneracy/20260803-061736
```

The plotter writes to a new directory under `results/generated_figures/`.

Run the focused N=32 causal control that freezes SORp's nonlinear face
coefficients once per sweep, calibrates its omega, and compares it with the
existing SORh and SORp implementations:

```bash
uv run python -m tests.nonlinear_studies.compare_frozen_sorp
```

The experiment selects `p_unsplit_config.coefficient_mode=frozen_per_sweep`
through the canonical `p_unsplit` method and writes its data under
`results/nonlinear_studies/frozen_sorp/`.

The collector uses the existing `Config.compute_omega_opt()` formulas as
references. Omega selection is restricted to the first contiguous converged
branch. A study-only stagnation safeguard stops a candidate after at least
2,000 sweeps when the residual improves by less than 0.1% over the latest
1,000 sweeps.
