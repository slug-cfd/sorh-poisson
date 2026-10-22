# Nonlinear study figures

These figures summarize the completed studies in the neighboring directories.
`performance_scaling.png` and `degeneracy_results.png` are earlier renderings
of the data used for the thesis performance and coefficient-contrast figures.
The corresponding thesis-style figures can be regenerated with
`2D/diagnostics/plot_nonlinear_thesis_results.py`.

`initial_guess.png` records the original fixed-omega comparison and its
inaccurate causal title has been cropped. It shows that reusing the zero-guess
omega made the linear-in-x SORp run stagnate; it does not show intrinsic
instability. The follow-up `initial_sorp_optimized/20260810-173658` study found
that reoptimizing SORp from omega 2.68 to 2.66 produced convergence in 624
sweeps, versus 863 sweeps for the optimized zero guess.

Figure provenance:

- `omega_calibration.png`: `omega/20260803-040244/all_runs.csv`
- `h_cfl_decomposition.png`: `h_cfl/20260803-060108/h_cfl_comparison.csv`
- `degeneracy_results.png`: `degeneracy/20260803-061736/best_by_case.csv`
- `initial_guess.png`: `initial/20260803-063356/initial_guess_comparison.csv`
- `performance_scaling.png`: `performance/20260803-064210/performance_comparison.csv`
- `timestep_comparison.png`: `timestep/20260803-055929/best_by_case.csv`
- `coefficient_profiles.png`: analytic coefficient variants used by the
  degeneracy study
