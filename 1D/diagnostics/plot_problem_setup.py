import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BASE_DIR / "results" / "generated_report_figures" / "problem_setup"

NONLINEAR_COLOR = "#007C78"
LINEAR_COLOR = "#252525"
AXIS_COLOR = "#4B5563"
GRID_COLOR = "#D9DEE5"
WEAK_REGION_END = 0.12


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "axes.edgecolor": AXIS_COLOR,
            "axes.linewidth": 0.9,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.75,
            "grid.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def nonlinear_steady_solution(x_values):
    coefficient = 1.0 - math.log(2.0)

    def solve_at_x(x_value):
        if x_value <= 0.0:
            return 0.0
        if x_value >= 1.0:
            return 1.0
        return brentq(
            lambda u_value: u_value - math.log1p(u_value) - coefficient * x_value,
            0.0,
            1.0,
            xtol=1.0e-14,
            rtol=1.0e-14,
        )

    solution = np.array([solve_at_x(float(x_value)) for x_value in x_values])
    residual = solution - np.log1p(solution) - coefficient * x_values
    if np.max(np.abs(residual)) > 1.0e-12:
        raise RuntimeError("Nonlinear steady-state inversion failed its residual check.")
    return solution


def diffusion_coefficient(solution):
    return solution / (1.0 + solution)


def format_axis(axis):
    axis.set_xlim(0.0, 1.0)
    axis.set_xticks(np.linspace(0.0, 1.0, 5))
    axis.grid(True, which="major")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_steady_solutions(x_values, nonlinear_solution):
    figure, axis = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    axis.axvspan(0.0, WEAK_REGION_END, color=NONLINEAR_COLOR, alpha=0.09, linewidth=0)
    axis.plot(
        x_values,
        nonlinear_solution,
        color=NONLINEAR_COLOR,
        linewidth=2.8,
        label="Nonlinear steady solution",
    )
    axis.plot(
        x_values,
        x_values,
        color=LINEAR_COLOR,
        linewidth=2.4,
        linestyle=":",
        label=r"Linear steady solution, $u=x$",
    )

    annotation_x = 0.035
    annotation_y = np.interp(annotation_x, x_values, nonlinear_solution)
    axis.text(
        0.29,
        0.18,
        r"Weak-diffusion edge: $\nu(u_{\mathrm{ss}}(x))\rightarrow 0$ as $x\rightarrow 0$",
        color=NONLINEAR_COLOR,
        fontsize=12,
        fontweight="semibold",
    )
    axis.annotate(
        "",
        xy=(annotation_x, annotation_y),
        xytext=(0.282, 0.195),
        arrowprops={
            "arrowstyle": "->",
            "color": NONLINEAR_COLOR,
            "linewidth": 1.6,
            "shrinkA": 4,
            "shrinkB": 3,
        },
    )

    axis.set_ylim(0.0, 1.02)
    axis.set_yticks(np.linspace(0.0, 1.0, 5))
    axis.set_xlabel(r"Position, $x$")
    axis.set_ylabel(r"Steady solution, $u_{\mathrm{ss}}(x)$")
    axis.legend(loc="lower right", frameon=False)
    format_axis(axis)
    return figure


def plot_diffusion_profile(x_values, nonlinear_solution):
    coefficient = diffusion_coefficient(nonlinear_solution)
    figure, axis = plt.subplots(figsize=(8.0, 4.3), constrained_layout=True)
    axis.axvspan(0.0, WEAK_REGION_END, color=NONLINEAR_COLOR, alpha=0.09, linewidth=0)
    axis.plot(
        x_values,
        coefficient,
        color=NONLINEAR_COLOR,
        linewidth=2.8,
    )

    annotation_x = 0.025
    annotation_y = np.interp(annotation_x, x_values, coefficient)
    axis.annotate(
        r"$\nu(u_{\mathrm{ss}}(x))\rightarrow 0$ as $x\rightarrow 0$",
        xy=(annotation_x, annotation_y),
        xytext=(0.27, 0.11),
        color=NONLINEAR_COLOR,
        fontsize=12,
        fontweight="semibold",
        arrowprops={
            "arrowstyle": "->",
            "color": NONLINEAR_COLOR,
            "linewidth": 1.6,
            "shrinkA": 4,
            "shrinkB": 3,
        },
    )

    axis.set_ylim(0.0, 0.52)
    axis.set_yticks(np.linspace(0.0, 0.5, 6))
    axis.set_xlabel(r"Position, $x$")
    axis.set_ylabel(r"Diffusion coefficient, $\nu(u_{\mathrm{ss}}(x))$")
    format_axis(axis)
    return figure


def save_figure(figure, output_dir, stem, overwrite):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
    figure.savefig(paths[0], dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(paths[1], bbox_inches="tight", facecolor="white")
    return paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the 1D steady solutions and nonlinear diffusion profile."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=("solutions", "coefficient"),
        default=("solutions", "coefficient"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_style()
    x_values = np.linspace(0.0, 1.0, 501)
    nonlinear_solution = nonlinear_steady_solution(x_values)

    figures = []
    if "solutions" in args.figures:
        figures.append(
            (
                plot_steady_solutions(x_values, nonlinear_solution),
                "one_dimensional_steady_solutions",
            )
        )
    if "coefficient" in args.figures:
        figures.append(
            (
                plot_diffusion_profile(x_values, nonlinear_solution),
                "one_dimensional_diffusion_profile",
            )
        )

    written = []
    for figure, stem in figures:
        written.extend(save_figure(figure, args.output_dir, stem, args.overwrite))
        plt.close(figure)

    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
