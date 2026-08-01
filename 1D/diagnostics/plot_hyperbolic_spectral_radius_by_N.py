import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sp_analysis_hyperbolic import spectral_radius as hyperbolic_spectral_radius


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"


def _rho_curve(N, r, omega_values):
    rho_values = np.full_like(omega_values, np.nan)
    for i, omega in enumerate(omega_values):
        if omega <= 0.0:
            continue
        rho, _ = hyperbolic_spectral_radius(N, r, omega)
        rho_values[i] = rho
    return rho_values


def _minimum_label(N, omega_values, rho_values):
    index = int(np.nanargmin(rho_values))
    return rf"$N={N}$ ($\rho_{{\min}}={rho_values[index]:.3f}$ at $\omega={omega_values[index]:.3f}$)"


def plot_hyperbolic_spectral_radius_by_N(
    N_values=(16, 32, 64, 128),
    xa=0.0,
    xb=1.0,
    cfl=0.8,
    v=1.0,
    omega_min=0.0,
    omega_max=1.5,
    samples=1501,
    output_stem=RESULTS_DIR / "hyperbolic_spectral_radius_by_N",
):
    representative_N = max(N_values)
    dx = (xb - xa) / representative_N
    dt = cfl * dx / v
    r = v * dt / dx

    omega_values = np.linspace(omega_min, omega_max, samples)
    rho_by_N = {
        N: _rho_curve(N, r, omega_values)
        for N in N_values
    }

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.15, top=0.88)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    linestyles = [
        "solid",
        (0, (6, 2)),
        (0, (4, 2, 1.5, 2)),
        (0, (1.5, 2)),
    ]

    for color, linestyle, N in zip(colors, linestyles, N_values):
        rho_values = rho_by_N[N]
        ax.plot(
            omega_values,
            rho_values,
            color=color,
            linewidth=2.0,
            linestyle=linestyle,
            label=_minimum_label(N, omega_values, rho_values),
        )

    ax.axhline(
        1.0,
        color="0.35",
        linewidth=1.2,
        linestyle=":",
        label=r"Stability threshold, $\rho(G)=1$",
    )

    ax.set_xlim(omega_min, omega_max)
    ax.set_xlabel(r"Relaxation parameter, $\omega$")
    ax.set_ylabel("Spectral radius,\n" + r"$\rho(G)$")
    ax.set_title("1D Hyperbolic SOR Spectral Radius by Grid Size")
    ax.grid(True, which="major", color="0.86", linewidth=0.8)
    ax.grid(True, which="minor", color="0.93", linewidth=0.5)
    ax.minorticks_on()
    ax.legend(loc="best", frameon=True, framealpha=0.95, edgecolor="0.8")

    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches=None)
    fig.savefig(pdf_path, bbox_inches=None)
    plt.close(fig)

    return {
        "N_values": tuple(N_values),
        "cfl": cfl,
        "r": r,
        "omega_min": omega_min,
        "omega_max": omega_max,
        "omega_values": omega_values,
        "rho_by_N": rho_by_N,
        "png_path": png_path,
        "pdf_path": pdf_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a hyperbolic 1D spectral-radius overlay plot for several N."
    )
    parser.add_argument("--N-values", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--xa", type=float, default=0.0)
    parser.add_argument("--xb", type=float, default=1.0)
    parser.add_argument("--cfl", type=float, default=0.8)
    parser.add_argument("--v", type=float, default=1.0)
    parser.add_argument("--omega-min", type=float, default=0.0)
    parser.add_argument("--omega-max", type=float, default=1.5)
    parser.add_argument("--samples", type=int, default=1501)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=RESULTS_DIR / "hyperbolic_spectral_radius_by_N",
        help="Output path without extension. PNG and PDF are both written.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = plot_hyperbolic_spectral_radius_by_N(
        N_values=args.N_values,
        xa=args.xa,
        xb=args.xb,
        cfl=args.cfl,
        v=args.v,
        omega_min=args.omega_min,
        omega_max=args.omega_max,
        samples=args.samples,
        output_stem=args.output_stem,
    )
    print(f"N values = {result['N_values']}")
    print(f"CFL = {result['cfl']:.12g}")
    print(f"hyperbolic r = v*dt/dx = {result['r']:.12g}")
    print(f"omega range = [{result['omega_min']:.12g}, {result['omega_max']:.12g}]")
    print(f"Wrote {result['png_path']}")
    print(f"Wrote {result['pdf_path']}")
