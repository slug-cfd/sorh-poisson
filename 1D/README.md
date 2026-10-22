# One-dimensional experiments

This workspace compares SORh, SORp, Gauss–Seidel, and Jacobi methods on linear and nonlinear one-dimensional problems. It includes configuration-driven runs, relaxation and spectral-radius analysis, figure collectors, and the preserved data used for the thesis report.

## Layout

| Path | Purpose |
| --- | --- |
| `main.py`, `solver1D.py` | Run a configured simulation and write its record. |
| `config1D.py`, `load_config1D.py`, `configs/` | Settings, JSON loading, and reusable configurations. |
| `methods/`, `problem_helpers.py` | Linear/nonlinear solvers and problem helpers. |
| `tests/` | Checks, benchmarks, sweeps, and figure-data collectors. |
| `diagnostics/` | Solution, report, and spectral plots. |
| `results/` | Preserved thesis data and generated outputs. |

## Run a case

Complete the [repository setup](../README.md#setup), then enter `1D/` from the repository root:

```bash
cd 1D
python main.py
python load_config1D.py
```

The default is `configs/config_sample1D.json`. Select another configuration with `python main.py --config configs/mixed_mode_sorp.json`. The driver writes `results/latest_run.json` by default, replacing the previous run there; use `--output` to preserve a separate record. The sample config enables plotting, so disable `plot_enabled` in the JSON for a noninteractive run.

## Results and reproduction

[`results/thesis_report_data/`](results/thesis_report_data/) is the frozen input for the completed 1D thesis figures. The report plotter reads it by default and writes figures to ignored `results/generated_report_figures/`:

```bash
python diagnostics/plot_report_figures.py --help
python diagnostics/plot_problem_setup.py --help
python diagnostics/plot_spectral_radius_overlay.py --help
```

To collect new data for Figures 1–4 in one run, including shared omega optimization, use:

```bash
python tests/collect_report_figure_data.py --figures 1 2 3 4
```

Each run gets its own directory under `results/generated_report_data/comprehensive/`. For smaller runs, the individual collectors can reuse the Figure 3 omega sweep:

```bash
python tests/collect_figure3_data.py
python tests/collect_figure1_data.py --omega-data results/generated_report_data/figure3_omega_sweeps.csv
python tests/collect_figure2_data.py --omega-data results/generated_report_data/figure3_omega_sweeps.csv
python tests/collect_figure4_data.py --omega-data results/generated_report_data/figure3_omega_sweeps.csv
```

Individual collectors write to `results/generated_report_data/` and overwrite their default CSVs on rerun; pass `--output` to keep separate files. When plotting newly collected data, give `plot_report_figures.py` the appropriate data-directory options. `results/reference/` contains a small representative run, while `results/studies/` and `results/archive/` hold other experiments locally.

These commands also work with `uv run python` in place of `python`.
