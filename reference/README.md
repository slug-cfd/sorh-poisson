# Reference material

This folder collects reader-facing material that accompanies the solver code and thesis results:

- `SORh Thesis Presentation - Final.pdf`: the final thesis presentation.
- `figures/thesis_figures/`: selected thesis figures, organized by chapter.
- `figures/other/`: additional plots useful for explaining the problems and results.
- `reports/`: supporting technical reports.
  - `nonlinear_sor_parameter_studies_report.pdf`: discussion of nonlinear SORh vs. SORp numerical studies. The most important finding here is the degeneracy and coefficient-contrast mechanism that leads to the performance difference discussed in the thesis.
  - `sorh_generalized_sor_report.pdf`: an in-depth analysis showing the relationship between SORh and Classical SOR. The AOR connection is not discussed here.  
- `wavefront/`: supporting materials for the design of *wavefront mode*, a parallelized implementation of SORh. Includes sweep-mode figures, a wavefront visualization, and a block-versus-wavefront comparison. This material provides context beyond the final thesis content.

The numerical data behind the solver studies are described in the [1D results guide](../1D/README.md#results-and-reproduction) and [2D results guide](../2D/results/README.md).
