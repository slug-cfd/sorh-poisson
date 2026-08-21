"""Spectral analysis for the default 2D SORh discretization.

The matrix coefficients follow the handwritten 2D Analysis 3 PDF:

    u_tx + u_ty = u_xx + u_yy - f

with backward-in-x/backward-in-y hyperbolic terms, centered old-time
Laplacian terms, and pointwise SOR relaxation.  The unknown vector uses the
PDF ordering

    k = i + Nx * (j - 1)

where ``Nx`` is the number of interior unknowns in the matrix.  This matches
the repository's Poisson configuration: ``config.Nx`` and ``config.Ny`` are
interior counts, while the stored arrays also include fixed boundary/ghost
layers.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "spectral_analysis"


@dataclass(frozen=True)
class SorhGrid:
    """Grid parameters matching the 2D Poisson interior-count convention."""

    Nx: int
    Ny: int
    xa: float
    xb: float
    ya: float
    yb: float
    cfl: float
    dt_factor: float
    ngc: int
    nx_unknowns: int
    ny_unknowns: int
    dx: float
    dy: float
    dt: float
    c1: float


@dataclass(frozen=True)
class SorhCoefficients:
    """L and R coefficients for L u^{n+1} = R u^n + forcing."""

    lower_x_new: float
    lower_y_new: float
    center_old: float
    lower_x_old: float
    upper_x_old: float
    lower_y_old: float
    upper_y_old: float


def make_poisson_grid(
    Nx: int,
    Ny: int,
    *,
    xa: float = 0.0,
    xb: float = 1.0,
    ya: float = 0.0,
    yb: float = 1.0,
    cfl: float = 0.8,
    dt_factor: float = 1.0,
    ngc: int = 1,
) -> SorhGrid:
    """Return grid parameters using the 2D configuration grid convention."""
    if ngc != 1:
        raise ValueError("The PDF SORh stencil uses one boundary layer, so ngc must be 1.")
    if Nx < 1 or Ny < 1:
        raise ValueError(
            f"Nx={Nx}, Ny={Ny} are invalid; both interior counts must be positive."
        )

    dx = (xb - xa) / (Nx + 1)
    dy = (yb - ya) / (Ny + 1)
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError(f"Grid spacings must be positive, got dx={dx}, dy={dy}.")

    dt = dt_factor * cfl * min(dx, dy)

    return SorhGrid(
        Nx=Nx,
        Ny=Ny,
        xa=xa,
        xb=xb,
        ya=ya,
        yb=yb,
        cfl=cfl,
        dt_factor=dt_factor,
        ngc=ngc,
        nx_unknowns=Nx,
        ny_unknowns=Ny,
        dx=dx,
        dy=dy,
        dt=dt,
        c1=1.0 / (dx + dy),
    )


def pdf_order_index(i: int, j: int, nx_unknowns: int) -> int:
    """Zero-based vector index for PDF one-based interior indices i, j."""
    return (i - 1) + nx_unknowns * (j - 1)


def sorh_coefficients(grid: SorhGrid, omega: float) -> SorhCoefficients:
    """Return the matrix coefficients from the PDF's rearranged SORh update."""
    dx, dy, dt, c1 = grid.dx, grid.dy, grid.dt, grid.c1

    lower_x_new = -omega * c1 * dy
    lower_y_new = -omega * c1 * dx

    center_old = 1.0 - (2.0 * omega * c1 * dt * ((dy / dx) + (dx / dy)))
    lower_x_old = (-omega * c1 * dy) + (omega * c1 * dt * (dy / dx))
    upper_x_old = omega * c1 * dt * (dy / dx)
    lower_y_old = (-omega * c1 * dx) + (omega * c1 * dt * (dx / dy))
    upper_y_old = omega * c1 * dt * (dx / dy)

    return SorhCoefficients(
        lower_x_new=lower_x_new,
        lower_y_new=lower_y_new,
        center_old=center_old,
        lower_x_old=lower_x_old,
        upper_x_old=upper_x_old,
        lower_y_old=lower_y_old,
        upper_y_old=upper_y_old,
    )


def sorh_matrices(grid: SorhGrid, omega: float) -> tuple[np.ndarray, np.ndarray]:
    """Build L and R for the PDF SORh ordering."""
    coeffs = sorh_coefficients(grid, omega)
    nx = grid.nx_unknowns
    ny = grid.ny_unknowns
    size = nx * ny

    L = np.eye(size)
    R = np.zeros((size, size))

    for j in range(1, ny + 1):
        for i in range(1, nx + 1):
            k = pdf_order_index(i, j, nx)

            R[k, k] = coeffs.center_old

            if i > 1:
                km = pdf_order_index(i - 1, j, nx)
                L[k, km] = coeffs.lower_x_new
                R[k, km] = coeffs.lower_x_old

            if i < nx:
                kp = pdf_order_index(i + 1, j, nx)
                R[k, kp] = coeffs.upper_x_old

            if j > 1:
                km = pdf_order_index(i, j - 1, nx)
                L[k, km] = coeffs.lower_y_new
                R[k, km] = coeffs.lower_y_old

            if j < ny:
                kp = pdf_order_index(i, j + 1, nx)
                R[k, kp] = coeffs.upper_y_old

    return L, R


def sorh_iteration_matrix(grid: SorhGrid, omega: float) -> np.ndarray:
    """Return G = L^{-1} R for one homogeneous SORh sweep."""
    L, R = sorh_matrices(grid, omega)
    return np.linalg.solve(L, R)


def spectral_radius(grid: SorhGrid, omega: float) -> tuple[float, np.ndarray]:
    """Compute the spectral radius of the PDF SORh iteration matrix."""
    G = sorh_iteration_matrix(grid, omega)
    eigvals = np.linalg.eigvals(G)
    return float(np.max(np.abs(eigvals))), eigvals


def _vector_to_grid(vec: np.ndarray, grid: SorhGrid) -> np.ndarray:
    values = np.zeros((grid.nx_unknowns, grid.ny_unknowns))
    for j in range(1, grid.ny_unknowns + 1):
        for i in range(1, grid.nx_unknowns + 1):
            values[i - 1, j - 1] = vec[pdf_order_index(i, j, grid.nx_unknowns)]
    return values


def _grid_to_vector(values: np.ndarray, grid: SorhGrid) -> np.ndarray:
    vec = np.zeros(grid.nx_unknowns * grid.ny_unknowns)
    for j in range(1, grid.ny_unknowns + 1):
        for i in range(1, grid.nx_unknowns + 1):
            vec[pdf_order_index(i, j, grid.nx_unknowns)] = values[i - 1, j - 1]
    return vec


def direct_pdf_sorh_sweep(vec: np.ndarray, grid: SorhGrid, omega: float) -> np.ndarray:
    """Apply one homogeneous PDF SORh sweep by loops for consistency checks."""
    old = _vector_to_grid(vec, grid)
    new = np.zeros_like(old)

    dx, dy, dt, c1 = grid.dx, grid.dy, grid.dt, grid.c1

    for j in range(grid.ny_unknowns):
        for i in range(grid.nx_unknowns):
            u_ij = old[i, j]
            old_left = old[i - 1, j] if i > 0 else 0.0
            old_right = old[i + 1, j] if i + 1 < grid.nx_unknowns else 0.0
            old_down = old[i, j - 1] if j > 0 else 0.0
            old_up = old[i, j + 1] if j + 1 < grid.ny_unknowns else 0.0

            new_left = new[i - 1, j] if i > 0 else 0.0
            new_down = new[i, j - 1] if j > 0 else 0.0

            laplacian = (
                (old_right - (2.0 * u_ij) + old_left) / (dx * dx)
                + (old_up - (2.0 * u_ij) + old_down) / (dy * dy)
            )
            delta = c1 * (
                dy * (new_left - old_left)
                + dx * (new_down - old_down)
                + dt * dx * dy * laplacian
            )
            new[i, j] = u_ij + (omega * delta)

    return _grid_to_vector(new, grid)


def consistency_error(grid: SorhGrid, omega: float) -> float:
    """Compare the analytic matrix with this analysis's direct PDF sweep."""
    rng = np.random.default_rng(12345)
    vec = rng.normal(size=grid.nx_unknowns * grid.ny_unknowns)
    G = sorh_iteration_matrix(grid, omega)
    matrix_next = G @ vec
    direct_next = direct_pdf_sorh_sweep(vec, grid, omega)
    return float(np.linalg.norm(matrix_next - direct_next, ord=np.inf))


def rho_curve(grid: SorhGrid, omega_values: np.ndarray) -> np.ndarray:
    rho_values = np.empty_like(omega_values, dtype=float)
    for index, omega in enumerate(omega_values):
        rho_values[index], _ = spectral_radius(grid, float(omega))
    return rho_values


def _plot_single_curve(
    grid: SorhGrid,
    omega_values: np.ndarray,
    rho_values: np.ndarray,
    outpath: Path,
) -> None:
    best = int(np.argmin(rho_values))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(omega_values, rho_values, linewidth=2)
    ax.scatter([omega_values[best]], [rho_values[best]], s=28, zorder=3)
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1)
    ax.grid(True, which="both", alpha=0.35)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho(G)$")
    ax.set_title(
        "2D SORh Spectral Radius "
        rf"($N_x=N_y={grid.Nx}$ interior, "
        rf"{grid.nx_unknowns}x{grid.ny_unknowns} unknowns)"
    )
    ax.annotate(
        rf"$\rho_{{min}}={rho_values[best]:.6g}$ at $\omega={omega_values[best]:.4g}$",
        xy=(omega_values[best], rho_values[best]),
        xytext=(10, 12),
        textcoords="offset points",
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=400)
    plt.close(fig)


def plot_omega_vs_rho(
    Ns: list[int],
    *,
    cfl: float = 0.8,
    dt_factor: float = 1.0,
    omega_min: float = 0.01,
    omega_max: float = 1.99,
    domega: float = 0.005,
    xa: float = 0.0,
    xb: float = 1.0,
    ya: float = 0.0,
    yb: float = 1.0,
    outdir: Path = RESULTS_DIR,
    make_single_plots: bool = True,
) -> dict:
    """Generate omega-vs-rho curves for square interior grids."""
    outdir.mkdir(parents=True, exist_ok=True)

    omega_values = np.arange(omega_min, omega_max + 0.5 * domega, domega)
    curves = {}
    rows = []

    fig, ax = plt.subplots(figsize=(8, 6))

    for N in Ns:
        grid = make_poisson_grid(
            N,
            N,
            xa=xa,
            xb=xb,
            ya=ya,
            yb=yb,
            cfl=cfl,
            dt_factor=dt_factor,
        )
        rho_values = rho_curve(grid, omega_values)
        best = int(np.argmin(rho_values))

        curves[N] = {
            "grid": grid,
            "rho_values": rho_values,
            "best_omega": float(omega_values[best]),
            "best_rho": float(rho_values[best]),
            "consistency_error": consistency_error(grid, float(omega_values[best])),
        }

        ax.plot(
            omega_values,
            rho_values,
            linewidth=2,
            label=(
                rf"$N_x=N_y={N}$ interior "
                rf"($\rho_{{min}}={rho_values[best]:.4g}$ at "
                rf"$\omega={omega_values[best]:.4g}$)"
            ),
        )
        ax.scatter([omega_values[best]], [rho_values[best]], s=24, zorder=3)

        for omega, rho in zip(omega_values, rho_values, strict=True):
            rows.append(
                {
                    "Nx": N,
                    "Ny": N,
                    "nx_unknowns": grid.nx_unknowns,
                    "ny_unknowns": grid.ny_unknowns,
                    "omega": float(omega),
                    "rho": float(rho),
                }
            )

        if make_single_plots:
            _plot_single_curve(
                grid,
                omega_values,
                rho_values,
                outdir / f"sorh_pdf_omega_vs_rho_N{N}.png",
            )

    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1)
    ax.grid(True, which="both", alpha=0.35)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho(G)$")
    ax.set_title("2D SORh Spectral Radius from PDF Discretization")
    ax.legend()
    fig.tight_layout()
    overlay_path = outdir / "sorh_pdf_omega_vs_rho_overlay.png"
    fig.savefig(overlay_path, dpi=400)
    plt.close(fig)

    csv_path = outdir / "sorh_pdf_spectral_radius.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Nx", "Ny", "nx_unknowns", "ny_unknowns", "omega", "rho"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = outdir / "sorh_pdf_summary.json"
    summary = {
        "omega_min": omega_min,
        "omega_max": omega_max,
        "domega": domega,
        "cfl": cfl,
        "dt_factor": dt_factor,
        "domain": {"xa": xa, "xb": xb, "ya": ya, "yb": yb},
        "overlay_plot": str(overlay_path),
        "csv": str(csv_path),
        "runs": {
            str(N): {
                "grid": asdict(curves[N]["grid"]),
                "best_omega": curves[N]["best_omega"],
                "best_rho": curves[N]["best_rho"],
                "consistency_error": curves[N]["consistency_error"],
            }
            for N in Ns
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    return {
        "omega_values": omega_values,
        "curves": curves,
        "overlay_path": overlay_path,
        "csv_path": csv_path,
        "summary_path": summary_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 2D SORh omega-vs-spectral-radius plots."
    )
    parser.add_argument("--Ns", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--cfl", type=float, default=0.8)
    parser.add_argument("--dt-factor", type=float, default=1.0)
    parser.add_argument("--omega-min", type=float, default=0.01)
    parser.add_argument("--omega-max", type=float, default=1.99)
    parser.add_argument("--domega", type=float, default=0.005)
    parser.add_argument("--xa", type=float, default=0.0)
    parser.add_argument("--xb", type=float, default=1.0)
    parser.add_argument("--ya", type=float, default=0.0)
    parser.add_argument("--yb", type=float, default=1.0)
    parser.add_argument("--outdir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--no-single-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = plot_omega_vs_rho(
        args.Ns,
        cfl=args.cfl,
        dt_factor=args.dt_factor,
        omega_min=args.omega_min,
        omega_max=args.omega_max,
        domega=args.domega,
        xa=args.xa,
        xb=args.xb,
        ya=args.ya,
        yb=args.yb,
        outdir=args.outdir,
        make_single_plots=not args.no_single_plots,
    )

    print(f"wrote overlay plot: {result['overlay_path']}")
    print(f"wrote spectral data: {result['csv_path']}")
    print(f"wrote summary: {result['summary_path']}")
    for N, curve in result["curves"].items():
        print(
            f"Nx=Ny={N}: best omega={curve['best_omega']:.6g}, "
            f"rho={curve['best_rho']:.6g}, "
            f"matrix/direct error={curve['consistency_error']:.3e}"
        )


if __name__ == "__main__":
    main()
