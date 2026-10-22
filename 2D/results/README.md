# 2D results

The 2D numerical records selected for this repository are:

- [Two-blob studies](two_blob_studies/): calibrated omega, performance,
  accuracy, and solutions.
- [Nonlinear studies](nonlinear_studies/): parameter studies, controls, and
  derived figures.
- [Omega-formula checks](h_unsplit_omega_formula_validation/): square and
  rectangular grids.

The first two folders contain the numerical data for the two-blob and nonlinear
figures and tables in the thesis. The corresponding plot scripts are
[two-blob problem](../diagnostics/plot_two_blob_problem.py),
[two-blob results](../diagnostics/plot_two_blob_results.py), and
[nonlinear results](../diagnostics/plot_nonlinear_thesis_results.py).
The frozen omega selections include documented manual refinements; the raw
trials and provenance are retained with each study.
The thesis result plotters read these frozen study runs by default and write
new figures under ignored `generated_figures/`, leaving the records unchanged.
Pass explicit data directories to plot a new experiment.
New omega-formula validation runs write under ignored
`generated_formula_validation/`.
