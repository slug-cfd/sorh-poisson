# Two-Blob Data Collection

These scripts collect the SORh/SORp data for the two-dimensional two-blob
Poisson problem. Shared defaults are stored in `study_defaults.json`:

- interior grids: `N = 16, 32, 64, 128, 256` in each direction;
- CFL values: `1.0` and `0.8`;
- lexicographic SORh (`mode = default`) and lexicographic SORp;
- performance equation-residual tolerance `1e-5`;
- performance update-norm stagnation safeguard `1e-10`;
- accuracy equation-residual tolerance `1e-10`;
- accuracy update-norm stagnation safeguard `1e-13`;
- five timing repetitions per performance case.

## 1. Optimal omega

Run the optimizer separately from the performance collector:

```bash
cd 2D
uv run python -m tests.two_blob_studies.collect_omega
```

For each `(N, method, CFL)` case, the script evaluates the analytical formula
first. It then alternates fixed increments to the left and right. Each direction
closes after two consecutive candidates fail to reduce the iteration count, a
candidate fails to converge, or five increments have been tested. The selected
omega is the converged candidate with the fewest iterations; distance from the
formula and runtime are used only as tie breakers.

The default output is
`results/two_blob_studies/omega/<timestamp>/` and contains:

- `omega_trials.csv`: every tested local candidate;
- `best_omega.csv`: one selected omega per `(N, method, CFL)`;
- `metadata/global_config.json`: all fixed solver, discretization, problem, and
  method parameters.

Both CSV files are checkpointed throughout the search. To resume an interrupted
search, provide its existing directory explicitly:

```bash
uv run python -m tests.two_blob_studies.collect_omega \
  --output-dir results/two_blob_studies/omega/RUN_TIMESTAMP --resume
```

Replace `RUN_TIMESTAMP` with the interrupted run's directory name.

## 2. Performance

After the omega search finishes, run:

```bash
uv run python -m tests.two_blob_studies.collect_performance
```

If `--omega-results` is omitted, the newest completed `best_omega.csv` is used.
The performance script alternates which method is timed first on successive
repetitions. Configuration construction is outside the timed region.

The default output is
`results/two_blob_studies/performance/<timestamp>/` and contains:

- `performance_runs.csv`: raw repeated iteration and timing results;
- `performance_summary.csv`: means, sample standard deviations, and ranges;
- `method_comparison.csv`: direct iteration and runtime ratios for SORp/SORh;
- `metadata/global_config.json`: shared parameters and the exact omega-result
  source used by the run.

Raw performance data are checkpointed after every repetition. The performance
collector supports the same `--output-dir ... --resume` pattern.

## 3. Accuracy

Accuracy is collected separately so practical performance convergence does not
contaminate the spatial-error results:

```bash
uv run python -m tests.two_blob_studies.collect_accuracy
```

Each case is solved once with residual tolerance `1e-10`, update-stagnation
threshold `1e-13`, and at most 100,000 iterations. A row is accepted for the
observed-order calculation only if:

1. the solver stops with `reason = tol_residual`;
2. the independently recomputed RMS equation residual is below `1e-10`; and
3. `residual_rms / discrete_lambda_min` is no more than `1e-3` of the measured
   RMS solution error.

The third condition bounds the remaining algebraic error using the smallest
eigenvalue magnitude of the discrete Dirichlet Poisson operator. It ensures the
reported continuum-solution error is dominated by spatial discretization rather
than incomplete iterative convergence.

The accuracy output is
`results/two_blob_studies/accuracy/<timestamp>/` and contains:

- `accuracy.csv`: L1, RMS, and infinity norms of solution error, final equation
  residual, and analytic-solution discrete truncation error, plus the algebraic
  error bound and validation result;
- `observed_orders.csv`: calculated only from validated rows;
- `method_accuracy_comparison.csv`: SORh/SORp error values and direct differences
  between their saved discrete solutions;
- `solutions/*.npz`: interior discrete solutions used for comparison and resume;
- `metadata/global_config.json`: the stricter accuracy convergence policy and
  all other fixed settings.

Accuracy rows and interior solutions are checkpointed after each case, and the
collector supports `--output-dir ... --resume`.

Edit `study_defaults.json` to change the full study configuration. `--Ns`,
`--cfls`, and `--repetitions` provide temporary command-line overrides without
changing the stored defaults. The accuracy collector accepts `--Ns` and
`--cfls`; its stricter convergence settings are kept in the `accuracy` block of
the JSON file.
