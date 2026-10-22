"""Create thesis-ready figures for the 2D nonlinear SORh/SORp results.

The script reads only the completed deterministic study outputs.  It writes
vector PDF figures for LaTeX and high-resolution PNG copies for visual review.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "2D" / "results" / "nonlinear_studies"
DEFAULT_OUTPUT = ROOT / "2D" / "results" / "generated_figures" / "nonlinear"

DEFAULT_PERFORMANCE_DIR = RESULTS / "performance" / "20260803-064210"
DEFAULT_DEGENERACY_DIR = RESULTS / "degeneracy" / "20260803-061736"

SORH_COLOR = "#3D8DFF"
SORP_COLOR = "#E87521"
RATIO_COLOR = "#007C78"
RATIO_EDGE_COLOR = "#005E5B"
REFERENCE_COLOR = "#252525"
GRID_COLOR = "#D9DEE5"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save_figure(
    figure: plt.Figure,
    stem: str,
    output_dir: Path,
    dpi: int,
    overwrite: bool,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"]
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite to replace it.")
    figure.savefig(
        paths[0],
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
        metadata={"Title": stem.replace("_", " ")},
    )
    figure.savefig(
        paths[1],
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )
    plt.close(figure)
    return paths


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "mathtext.fontset": "dejavusans",
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "axes.edgecolor": "#4B5563",
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.65,
            "grid.linewidth": 0.55,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 2.25,
            "lines.markersize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def performance_figure(
    performance_csv: Path,
    output_dir: Path,
    dpi: int,
    overwrite: bool,
) -> list[Path]:
    rows = sorted(read_csv(performance_csv), key=lambda row: int(row["N"]))
    grid_sizes = np.asarray([int(row["N"]) for row in rows], dtype=float)
    sorh_sweeps = np.asarray([int(row["sorh_sweeps"]) for row in rows], dtype=float)
    sorp_sweeps = np.asarray([int(row["sorp_sweeps"]) for row in rows], dtype=float)
    speedup = sorp_sweeps / sorh_sweeps

    sorh_exponent = np.polyfit(np.log(grid_sizes), np.log(sorh_sweeps), 1)[0]
    sorp_exponent = np.polyfit(np.log(grid_sizes), np.log(sorp_sweeps), 1)[0]

    figure, axes = plt.subplots(1, 2, figsize=(6.45, 2.75), constrained_layout=True)

    axes[0].plot(
        grid_sizes,
        sorh_sweeps,
        marker="o",
        markerfacecolor="white",
        markeredgewidth=1.2,
        color=SORH_COLOR,
        label=rf"SORh: $N^{{{sorh_exponent:.2f}}}$",
    )
    axes[0].plot(
        grid_sizes,
        sorp_sweeps,
        marker="s",
        color=SORP_COLOR,
        label=rf"SORp: $N^{{{sorp_exponent:.2f}}}$",
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xticks(grid_sizes, [str(int(value)) for value in grid_sizes])
    axes[0].set_xlabel(r"Interior grid size, $N$")
    axes[0].set_ylabel("Iterations to tolerance")
    axes[0].set_title("(a) Empirical iteration scaling", loc="left", fontweight="bold")
    axes[0].legend(loc="upper left")

    bars = axes[1].bar(
        np.arange(len(grid_sizes)),
        speedup,
        width=0.58,
        color=RATIO_COLOR,
        edgecolor=RATIO_EDGE_COLOR,
        linewidth=0.8,
    )
    axes[1].axhline(1.0, color=REFERENCE_COLOR, linestyle="--", linewidth=1.0)
    axes[1].set_xticks(np.arange(len(grid_sizes)), [str(int(value)) for value in grid_sizes])
    axes[1].set_xlabel(r"Interior grid size, $N$")
    axes[1].set_ylabel("SORp iterations / SORh iterations")
    axes[1].set_title("(b) SORh iteration advantage", loc="left", fontweight="bold")
    axes[1].set_ylim(0.0, 5.9)
    for bar, value in zip(bars, speedup, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.12,
            rf"${value:.2f}\times$",
            ha="center",
            va="bottom",
            fontsize=8.2,
        )

    return save_figure(
        figure,
        "nonlinear_baseline_performance",
        output_dir,
        dpi,
        overwrite,
    )


def degeneracy_figure(
    degeneracy_csv: Path,
    output_dir: Path,
    dpi: int,
    overwrite: bool,
) -> list[Path]:
    rows = read_csv(degeneracy_csv)
    grouped: dict[float, dict[str, dict[str, str]]] = {}
    for row in rows:
        ratio = float(row["ellipticity_ratio"])
        grouped.setdefault(ratio, {})[row["method"]] = row

    ratios = np.asarray(sorted(grouped), dtype=float)
    positions = np.arange(len(ratios), dtype=float)
    sorh_rows = [grouped[ratio]["h_unsplit"] for ratio in ratios]
    sorp_rows = [grouped[ratio]["p_unsplit"] for ratio in ratios]
    sorh_sweeps = np.asarray([int(row["sweeps"]) for row in sorh_rows], dtype=float)
    sorp_sweeps = np.asarray([int(row["sweeps"]) for row in sorp_rows], dtype=float)
    speedup = sorp_sweeps / sorh_sweeps
    sorh_shift = np.asarray(
        [float(row["omega"]) / float(row["formula_reference"]) for row in sorh_rows]
    )
    sorp_shift = np.asarray(
        [float(row["omega"]) / float(row["formula_reference"]) for row in sorp_rows]
    )

    figure = plt.figure(figsize=(6.45, 5.25), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.05))
    count_axis = figure.add_subplot(grid[0, :])
    ratio_axis = figure.add_subplot(grid[1, 0])
    shift_axis = figure.add_subplot(grid[1, 1])
    axes = (count_axis, ratio_axis, shift_axis)

    count_axis.plot(
        positions,
        sorh_sweeps,
        marker="o",
        markerfacecolor="white",
        markeredgewidth=1.2,
        color=SORH_COLOR,
        label="SORh",
    )
    count_axis.plot(positions, sorp_sweeps, marker="s", color=SORP_COLOR, label="SORp")
    count_axis.set_yscale("log")
    count_axis.set_ylabel("Iterations to tolerance")
    count_axis.set_title("(a) Optimized iteration counts", loc="left", fontweight="bold")
    count_axis.legend(ncol=2, loc="upper right")

    ratio_axis.plot(positions, speedup, marker="o", color=RATIO_COLOR)
    ratio_axis.axhline(1.0, color=REFERENCE_COLOR, linestyle="--", linewidth=1.0)
    ratio_axis.set_ylabel("SORp iterations / SORh iterations")
    ratio_axis.set_title("(b) SORh iteration advantage", loc="left", fontweight="bold")
    ratio_axis.annotate(
        rf"${speedup[0]:.2f}\times$",
        (positions[0], speedup[0]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
    )
    ratio_axis.annotate(
        rf"${speedup[-1]:.2f}\times$",
        (positions[-1], speedup[-1]),
        xytext=(-4, 6),
        textcoords="offset points",
        ha="right",
        fontsize=8,
    )

    shift_axis.plot(
        positions,
        sorh_shift,
        marker="o",
        markerfacecolor="white",
        markeredgewidth=1.2,
        color=SORH_COLOR,
        label="SORh",
    )
    shift_axis.plot(positions, sorp_shift, marker="s", color=SORP_COLOR, label="SORp")
    shift_axis.axhline(1.0, color=REFERENCE_COLOR, linestyle="--", linewidth=1.0)
    shift_axis.set_ylabel(r"Selected $\omega$ / linear prediction")
    shift_axis.set_title("(c) Nonlinear relaxation shift", loc="left", fontweight="bold")
    tick_labels = ["0", ".002", ".020", ".167", ".667", "1"]
    for axis in axes:
        axis.set_xlim(-0.25, len(positions) - 0.75)
        axis.set_xticks(positions, tick_labels)
        axis.set_xlabel(r"Tested ellipticity ratio, $v_{\min}/v_{\max}$")
        axis.grid(True, which="major")

    return save_figure(
        figure,
        "nonlinear_coefficient_contrast",
        output_dir,
        dpi,
        overwrite,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--performance-dir",
        type=Path,
        default=DEFAULT_PERFORMANCE_DIR,
        help="directory containing performance_comparison.csv",
    )
    parser.add_argument(
        "--degeneracy-dir",
        type=Path,
        default=DEFAULT_DEGENERACY_DIR,
        help="directory containing best_by_case.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    made = []
    made.extend(
        performance_figure(
            args.performance_dir / "performance_comparison.csv",
            args.output_dir,
            args.dpi,
            args.overwrite,
        )
    )
    made.extend(
        degeneracy_figure(
            args.degeneracy_dir / "best_by_case.csv",
            args.output_dir,
            args.dpi,
            args.overwrite,
        )
    )
    for path in made:
        print(f"wrote {path.resolve()}")


if __name__ == "__main__":
    main()
