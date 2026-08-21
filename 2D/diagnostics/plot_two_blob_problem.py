import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


from configuration import initialize_simulation, load_config


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = BASE_DIR / "configs" / "config_poisson.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results" / "two_blob_studies" / "figures"


def plot_two_blob_problem(config, sample_points=401):
    """Plot the analytic source and exact potential for the two-blob problem."""
    if config.problem != "poisson" or config.problem_case != "two_blob":
        raise ValueError("Figure 1 requires the Poisson two_blob configuration.")

    x = np.linspace(config.xa, config.xb, sample_points)
    y = np.linspace(config.ya, config.yb, sample_points)
    X, Y = np.meshgrid(x, y, indexing="ij")
    source = config._eval_rhs(X, Y).T
    exact = config._eval_exact(X, Y).T

    source_min = float(np.min(source))
    source_max = float(np.max(source))
    negative_levels = np.linspace(source_min, 0.0, 6)[1:-1]
    positive_levels = np.linspace(0.0, source_max, 12)[1:-1]
    source_levels = np.concatenate((negative_levels, positive_levels))
    exact_levels = np.linspace(float(np.min(exact)), float(np.max(exact)), 17)[1:-1]

    style = {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.edgecolor": "#4B5563",
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(8.8, 3.7),
            constrained_layout=True,
        )

        source_plot = axes[0].contour(
            x,
            y,
            source,
            levels=source_levels,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=source_min, vcenter=0.0, vmax=source_max),
            linewidths=1.25,
            linestyles="solid",
        )
        axes[0].contour(
            x,
            y,
            source,
            levels=[0.0],
            colors="#374151",
            linewidths=0.7,
        )
        source_bar = fig.colorbar(source_plot, ax=axes[0], pad=0.03)
        source_bar.set_label(r"$f(x,y)$")

        exact_plot = axes[1].contour(
            x,
            y,
            exact,
            levels=exact_levels,
            cmap="viridis",
            linewidths=1.25,
            linestyles="solid",
        )
        exact_bar = fig.colorbar(exact_plot, ax=axes[1], pad=0.03)
        exact_bar.set_label(r"$\phi_{\mathrm{exact}}(x,y)$")

        axes[0].set_title(r"(a) Source $f(x,y)$")
        axes[1].set_title(r"(b) Exact potential $\phi_{\mathrm{exact}}(x,y)$")
        for ax in axes:
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([-20, -10, 0, 10, 20])
            ax.set_yticks([-20, -10, 0, 10, 20])

    return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create the report figure defining the two-blob Poisson problem."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="fig1-two-blob-problem")
    parser.add_argument("--sample-points", type=int, default=401)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = initialize_simulation(load_config(args.config))
    fig = plot_two_blob_problem(config, sample_points=args.sample_points)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        args.output_dir / f"{args.stem}.png",
        args.output_dir / f"{args.stem}.pdf",
    ]
    for path in output_paths:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite to replace it.")

    fig.savefig(output_paths[0], dpi=args.dpi, bbox_inches="tight")
    fig.savefig(output_paths[1], bbox_inches="tight")

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    for path in output_paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
