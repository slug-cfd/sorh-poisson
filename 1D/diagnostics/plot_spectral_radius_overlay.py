import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sp_analysis_hyperbolic import spectral_radius as hyperbolic_spectral_radius
from sp_analysis_parabolic import spectral_radius as parabolic_spectral_radius


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
SORH_COLOR = "#3D8DFF"
SORP_COLOR = "#E87521"
EXACT_COLOR = "#252525"


def _omega_values(omega_min, omega_max, samples):
    return np.linspace(omega_min, omega_max, samples)


def _rho_curve(spectral_radius_fn, N, r, omega_values):
    rho_values = np.empty_like(omega_values)
    for i, omega in enumerate(omega_values):
        rho, _ = spectral_radius_fn(N, r, omega)
        rho_values[i] = rho
    return rho_values


def _minimum_label(name, omega_values, rho_values):
    index = int(np.argmin(rho_values))
    name_prefix = f"{name} " if name else ""
    return (
        rf"{name_prefix}"
        rf"($\rho_{{\min}}={rho_values[index]:.3f}$ at $\omega={omega_values[index]:.3f}$)"
    )


def _validate_grid_size(N):
    interior_cells = N - 1
    if interior_cells < 1 or interior_cells & (interior_cells - 1):
        raise ValueError(
            "N must be one greater than a power of two "
            "(for example, 17, 33, 65, or 129)."
        )


def _set_method_axis_style(
    ax,
    omega_min,
    omega_max,
    rho_values,
    *,
    x_tick_count=6,
    legend_fontsize=None,
    y_limits=None,
):
    rho_min = float(np.min(rho_values))
    rho_max = float(np.max(rho_values))
    if y_limits is None:
        y_min = 0.96 * rho_min
        y_max = 1.03 * rho_max
    else:
        y_min, y_max = y_limits
    tick_candidates = np.array([0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    y_ticks = tick_candidates[(tick_candidates >= y_min) & (tick_candidates <= y_max)]
    x_ticks = np.linspace(omega_min, omega_max, x_tick_count)

    ax.set_xlim(omega_min, omega_max)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{tick:g}" for tick in x_ticks])
    ax.set_yscale("log")
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{tick:.1f}" for tick in y_ticks])
    ax.grid(True, which="major", color="0.86", linewidth=0.8)
    ax.grid(True, which="minor", color="0.93", linewidth=0.5)
    ax.minorticks_on()
    ax.legend(
        loc="best",
        frameon=True,
        framealpha=0.95,
        edgecolor="0.8",
        fontsize=legend_fontsize,
    )


def plot_spectral_radius_overlay(
    N=65,
    xa=0.0,
    xb=1.0,
    cfl=0.8,
    v=1.0,
    sorh_omega_min=0.5,
    sorh_omega_max=1.25,
    sorp_omega_min=1.5,
    sorp_omega_max=3.0,
    samples=2501,
    layout="rows",
    output_stem=None,
):
    _validate_grid_size(N)
    if layout not in {"rows", "columns"}:
        raise ValueError("layout must be either 'rows' or 'columns'.")

    dx = (xb - xa) / N

    dt_hyperbolic = cfl * dx / v
    r_hyperbolic = v * dt_hyperbolic / dx

    dt_parabolic = cfl * dx**2 / (2.0 * v)
    r_parabolic = v * dt_parabolic / dx**2

    sorh_omega_values = _omega_values(
        sorh_omega_min,
        sorh_omega_max,
        samples,
    )
    rho_hyperbolic = _rho_curve(
        hyperbolic_spectral_radius,
        N,
        r_hyperbolic,
        sorh_omega_values,
    )
    sorp_omega_values = _omega_values(
        sorp_omega_min,
        sorp_omega_max,
        samples,
    )
    rho_parabolic = _rho_curve(
        parabolic_spectral_radius,
        N,
        r_parabolic,
        sorp_omega_values,
    )

    if output_stem is None:
        filename_suffix = "_2column" if layout == "columns" else ""
        output_stem = (
            RESULTS_DIR
            / "generated_figures"
            / "spectral_radius"
            / f"spectral_radius_comparison_1D_N{N}{filename_suffix}"
        )
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    if layout == "columns":
        subplot_shape = {"nrows": 1, "ncols": 2, "figsize": (7.8, 3.8)}
        minimum_label_names = (None, None)
        x_tick_count = 4
        legend_fontsize = 8
        y_limits = (
            0.96 * min(np.min(rho_hyperbolic), np.min(rho_parabolic)),
            1.03 * max(np.max(rho_hyperbolic), np.max(rho_parabolic)),
        )
    else:
        subplot_shape = {"nrows": 2, "ncols": 1, "figsize": (7.2, 7.0)}
        minimum_label_names = ("SORh", "SORp")
        x_tick_count = 6
        legend_fontsize = None
        y_limits = None

    fig, axes = plt.subplots(
        **subplot_shape,
        constrained_layout=True,
    )
    ax_hyperbolic, ax_parabolic = np.asarray(axes).flat

    ax_hyperbolic.plot(
        sorh_omega_values,
        rho_hyperbolic,
        color=SORH_COLOR,
        linewidth=2.5,
        label=_minimum_label(
            minimum_label_names[0],
            sorh_omega_values,
            rho_hyperbolic,
        ),
    )
    ax_hyperbolic.axhline(
        1.0,
        color=EXACT_COLOR,
        linewidth=1.5,
        linestyle=":",
        label=r"Stability threshold, $\rho(G)=1$",
    )
    ax_hyperbolic.set_title("SORh")
    _set_method_axis_style(
        ax_hyperbolic,
        sorh_omega_min,
        sorh_omega_max,
        rho_hyperbolic,
        x_tick_count=x_tick_count,
        legend_fontsize=legend_fontsize,
        y_limits=y_limits,
    )

    ax_parabolic.plot(
        sorp_omega_values,
        rho_parabolic,
        color=SORP_COLOR,
        linewidth=2.5,
        label=_minimum_label(
            minimum_label_names[1],
            sorp_omega_values,
            rho_parabolic,
        ),
    )
    ax_parabolic.axhline(
        1.0,
        color=EXACT_COLOR,
        linewidth=1.5,
        linestyle=":",
        label=r"Stability threshold, $\rho(G)=1$",
    )
    ax_parabolic.set_title("SORp")
    _set_method_axis_style(
        ax_parabolic,
        sorp_omega_min,
        sorp_omega_max,
        rho_parabolic,
        x_tick_count=x_tick_count,
        legend_fontsize=legend_fontsize,
        y_limits=y_limits,
    )

    if layout == "columns":
        fig.supxlabel(r"Relaxation parameter, $\omega$")
    else:
        ax_parabolic.set_xlabel(r"Relaxation parameter, $\omega$")
    fig.supylabel(r"Spectral radius, $\rho(G)$ (log scale)")

    fig.suptitle(
        rf"1D SOR Spectral Radius vs. Relaxation Parameter "
        rf"($N_{{\mathrm{{int}}}}={N - 1}$)"
    )

    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    return {
        "N": N,
        "interior_cells": N - 1,
        "layout": layout,
        "dx": dx,
        "dt_hyperbolic": dt_hyperbolic,
        "r_hyperbolic": r_hyperbolic,
        "dt_parabolic": dt_parabolic,
        "r_parabolic": r_parabolic,
        "sorh_omega_values": sorh_omega_values,
        "sorp_omega_values": sorp_omega_values,
        "rho_hyperbolic": rho_hyperbolic,
        "rho_parabolic": rho_parabolic,
        "png_path": png_path,
        "pdf_path": pdf_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a report-ready two-panel 1D spectral-radius plot."
    )
    parser.add_argument("--N", type=int, default=65)
    parser.add_argument("--xa", type=float, default=0.0)
    parser.add_argument("--xb", type=float, default=1.0)
    parser.add_argument("--cfl", type=float, default=0.8)
    parser.add_argument("--v", type=float, default=1.0)
    parser.add_argument("--sorh-omega-min", type=float, default=0.5)
    parser.add_argument("--sorh-omega-max", type=float, default=1.25)
    parser.add_argument("--sorp-omega-min", type=float, default=1.5)
    parser.add_argument("--sorp-omega-max", type=float, default=3.0)
    parser.add_argument("--samples", type=int, default=2501)
    parser.add_argument(
        "--layout",
        choices=("rows", "columns"),
        default="rows",
        help="Arrange the SORh and SORp panels vertically or side by side.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=None,
        help=(
            "Output path without extension. By default, PNG and PDF are written "
            "under 1D/results/generated_figures/spectral_radius with N included "
            "in the filename."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = plot_spectral_radius_overlay(
        N=args.N,
        xa=args.xa,
        xb=args.xb,
        cfl=args.cfl,
        v=args.v,
        sorh_omega_min=args.sorh_omega_min,
        sorh_omega_max=args.sorh_omega_max,
        sorp_omega_min=args.sorp_omega_min,
        sorp_omega_max=args.sorp_omega_max,
        samples=args.samples,
        layout=args.layout,
        output_stem=args.output_stem,
    )
    print(f"N = {result['N']}")
    print(f"interior cells = {result['interior_cells']}")
    print(f"dx = {result['dx']:.12g}")
    print(f"hyperbolic dt = {result['dt_hyperbolic']:.12g}")
    print(f"hyperbolic r = v*dt/dx = {result['r_hyperbolic']:.12g}")
    print(f"parabolic dt = {result['dt_parabolic']:.12g}")
    print(f"parabolic r = v*dt/dx^2 = {result['r_parabolic']:.12g}")
    print(f"Wrote {result['png_path']}")
    print(f"Wrote {result['pdf_path']}")
