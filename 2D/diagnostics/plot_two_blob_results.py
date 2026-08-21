"""Create report-quality performance and accuracy figures for the two-blob study."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
STUDY_RESULTS = BASE_DIR / "results" / "two_blob_studies"
DEFAULT_OUTPUT_DIR = STUDY_RESULTS / "figures"

METHOD_ORDER = ("h_unsplit", "p_unsplit")
METHOD_STYLES = {
    "h_unsplit": {
        "label": "SORh",
        "color": "#3D8DFF",
        "marker": "o",
        "linestyle": "-",
        "markerfacecolor": "white",
    },
    "p_unsplit": {
        "label": "SORp",
        "color": "#E87521",
        "marker": "s",
        "linestyle": "-",
        "markerfacecolor": "#E87521",
    },
}

SOLUTION_ERROR_COLOR = "#007C78"
TRUNCATION_ERROR_COLOR = "#A64D79"
EXACT_COLOR = "#252525"

PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.edgecolor": "#4B5563",
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "savefig.transparent": False,
}


def latest_result_dir(study, required_files):
    root = STUDY_RESULTS / study
    matches = [
        path
        for path in sorted(root.glob("*"))
        if path.is_dir() and all((path / name).exists() for name in required_files)
    ]
    if not matches:
        required = ", ".join(required_files)
        raise FileNotFoundError(
            f"No completed {study} result directory contains: {required}"
        )
    return matches[-1]


def load_csv(path, required_columns):
    frame = pd.read_csv(path)
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return frame


def configure_grid_axis(ax, grid_sizes):
    ticks = sorted(int(value) for value in np.unique(grid_sizes))
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(value) for value in ticks])
    ax.set_xlabel(r"Interior grid size, $N$")


def configure_log_grid(ax):
    ax.grid(True, which="major", color="#D9DEE5", linewidth=0.8)
    ax.grid(True, which="minor", color="#ECEFF3", linewidth=0.5)


def plot_performance(summary, cfl):
    selected = summary[np.isclose(summary["cfl"], cfl)].copy()
    if selected.empty:
        raise ValueError(f"No performance rows were found for CFL={cfl:g}.")

    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(8.8, 3.7),
            constrained_layout=True,
        )

        for method in METHOD_ORDER:
            group = selected[selected["method"] == method].sort_values("N")
            if group.empty:
                raise ValueError(f"No performance rows were found for {method}.")
            style = METHOD_STYLES[method]
            common = {
                "label": style["label"],
                "color": style["color"],
                "marker": style["marker"],
                "linestyle": style["linestyle"],
                "markerfacecolor": style["markerfacecolor"],
                "markeredgecolor": style["color"],
                "markeredgewidth": 1.2,
                "linewidth": 2.25,
                "markersize": 6,
            }
            axes[0].plot(group["N"], group["iterations_mean"], **common)
            axes[1].plot(
                group["N"],
                group["wall_time_mean_sec"],
                **common,
            )

        for ax in axes:
            configure_grid_axis(ax, selected["N"])
            ax.set_yscale("log")
            configure_log_grid(ax)

        axes[0].set_title("(a) Iteration count")
        axes[0].set_ylabel("Iterations")
        axes[1].set_title("(b) Mean runtime")
        axes[1].set_ylabel("Runtime (s)")
        axes[1].legend(frameon=False, loc="upper left")
        figure.suptitle(rf"Two-Blob Poisson Performance ($\mathrm{{CFL}}={cfl:.1f}$)")

    return figure


def plot_combined_performance(summary, cfls=(1.0, 0.8)):
    selected = summary[
        np.logical_or.reduce(
            [np.isclose(summary["cfl"], cfl) for cfl in cfls]
        )
    ].copy()
    if selected.empty:
        raise ValueError("No performance rows were found for the requested CFL values.")

    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(8.8, 3.7),
            constrained_layout=True,
        )
        runtime_handles = {}
        combination_linestyles = {
            ("h_unsplit", 1.0): "-",
            ("p_unsplit", 1.0): (0, (6, 2, 1.5, 2)),
            ("h_unsplit", 0.8): (0, (6, 2)),
            ("p_unsplit", 0.8): (0, (1.5, 2)),
        }

        # Draw CFL=1 first. When curves coincide, the dashed CFL=0.8 line
        # remains visible over the solid line, while its larger open marker
        # surrounds the smaller filled CFL=1 marker.
        for method in METHOD_ORDER:
            method_style = METHOD_STYLES[method]
            for cfl in cfls:
                group = selected[
                    (selected["method"] == method)
                    & np.isclose(selected["cfl"], cfl)
                ].sort_values("N")
                if group.empty:
                    raise ValueError(
                        f"No performance rows were found for {method}, CFL={cfl:g}."
                    )

                is_reference = np.isclose(cfl, 1.0)
                label = rf"{method_style['label']}, $\mathrm{{CFL}}={cfl:.1f}$"
                common = {
                    "label": label,
                    "color": method_style["color"],
                    "marker": method_style["marker"],
                    "linestyle": combination_linestyles[
                        (method, round(float(cfl), 1))
                    ],
                    "linewidth": 2.25,
                    "markersize": 5.5 if is_reference else 7,
                    "markerfacecolor": (
                        method_style["color"] if is_reference else "none"
                    ),
                    "markeredgecolor": method_style["color"],
                    "markeredgewidth": 1.1,
                    "zorder": 2 if is_reference else 3,
                }
                axes[0].plot(
                    group["N"],
                    group["iterations_mean"],
                    **common,
                )
                handle, = axes[1].plot(
                    group["N"],
                    group["wall_time_mean_sec"],
                    **common,
                )
                runtime_handles[(method, round(float(cfl), 1))] = handle

        for ax in axes:
            configure_grid_axis(ax, selected["N"])
            ax.set_yscale("log")
            configure_log_grid(ax)

        axes[0].set_title("(a) Iteration count")
        axes[0].set_ylabel("Iterations")
        axes[1].set_title("(b) Mean runtime")
        axes[1].set_ylabel("Runtime (s)")

        legend_order = [
            ("h_unsplit", 1.0),
            ("p_unsplit", 1.0),
            ("h_unsplit", 0.8),
            ("p_unsplit", 0.8),
        ]
        handles = [runtime_handles[key] for key in legend_order]
        labels = [
            rf"{METHOD_STYLES[method]['label']}, $\mathrm{{CFL}}={cfl:.1f}$"
            for method, cfl in legend_order
        ]
        axes[1].legend(
            handles,
            labels,
            frameon=False,
            loc="upper left",
            handlelength=2.8,
        )
        figure.suptitle("Two-Blob Poisson Performance")

    return figure


def plot_accuracy(accuracy, observed_orders):
    valid = accuracy[
        (accuracy["accuracy_valid"].astype(str) == "True")
        & (accuracy["method"] == "p_unsplit")
        & np.isclose(accuracy["cfl"], 1.0)
    ].sort_values("N")
    if len(valid) < 3:
        raise ValueError("At least three validated representative accuracy rows are required.")

    orders = observed_orders[
        (observed_orders["method"] == "p_unsplit")
        & np.isclose(observed_orders["cfl"], 1.0)
    ].sort_values("N_fine")
    if orders.empty:
        raise ValueError("No representative observed-order rows were found.")

    anchor = valid[valid["N"] == 64]
    if anchor.empty:
        anchor = valid.iloc[[len(valid) // 2]]
    anchor = anchor.iloc[0]
    second_order = anchor["error_rms"] * (valid["dx"] / anchor["dx"]) ** 2

    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(8.8, 3.7),
            constrained_layout=True,
        )

        axes[0].plot(
            valid["N"],
            valid["error_rms"],
            color=SOLUTION_ERROR_COLOR,
            marker="o",
            linestyle="-",
            linewidth=2.25,
            markersize=6,
            label="Solution error",
        )
        axes[0].plot(
            valid["N"],
            valid["truncation_rms"],
            color=TRUNCATION_ERROR_COLOR,
            marker="s",
            linestyle="--",
            linewidth=2.25,
            markersize=6,
            label="Truncation error",
        )
        axes[0].plot(
            valid["N"],
            second_order,
            color=EXACT_COLOR,
            linestyle=":",
            linewidth=1.5,
            label="Second-order reference",
        )
        configure_grid_axis(axes[0], valid["N"])
        axes[0].set_yscale("log")
        axes[0].set_ylabel("RMS error")
        axes[0].set_title("(a) Error norms")
        configure_log_grid(axes[0])
        axes[0].legend(frameon=False, loc="lower left")

        axes[1].plot(
            orders["N_fine"],
            orders["solution_order_rms"],
            color=SOLUTION_ERROR_COLOR,
            marker="o",
            linestyle="-",
            linewidth=2.25,
            markersize=6,
            label="Solution error",
        )
        axes[1].plot(
            orders["N_fine"],
            orders["truncation_order_rms"],
            color=TRUNCATION_ERROR_COLOR,
            marker="s",
            linestyle="--",
            linewidth=2.25,
            markersize=6,
            label="Truncation error",
        )
        axes[1].axhline(
            2.0,
            color=EXACT_COLOR,
            linestyle=":",
            linewidth=1.5,
            label="Second order",
        )
        configure_grid_axis(axes[1], orders["N_fine"])
        axes[1].set_ylabel("Observed order")
        axes[1].set_ylim(bottom=0.0)
        axes[1].set_title("(b) Local RMS order")
        axes[1].grid(True, which="major", color="#D9DEE5", linewidth=0.8)
        axes[1].legend(frameon=False, loc="upper right")

        figure.suptitle(r"Two-Blob Poisson Accuracy (SORp, $\mathrm{CFL}=1.0$)")

    return figure


def save_figure(figure, output_dir, stem, dpi, overwrite):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite to replace it.")
    figure.savefig(paths[0], dpi=dpi, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    return paths


def cfl_stem(cfl):
    return f"{cfl:.1f}".replace(".", "p")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--performance-dir", type=Path)
    parser.add_argument("--accuracy-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--performance-cfls", type=float, nargs="+", default=[1.0, 0.8])
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    performance_dir = args.performance_dir or latest_result_dir(
        "performance", ["performance_summary.csv"]
    )
    accuracy_dir = args.accuracy_dir or latest_result_dir(
        "accuracy", ["accuracy.csv", "observed_orders.csv"]
    )

    performance = load_csv(
        performance_dir / "performance_summary.csv",
        {
            "N",
            "method",
            "cfl",
            "iterations_mean",
            "wall_time_mean_sec",
            "wall_time_std_sec",
        },
    )
    accuracy = load_csv(
        accuracy_dir / "accuracy.csv",
        {
            "N",
            "method",
            "cfl",
            "dx",
            "accuracy_valid",
            "error_rms",
            "truncation_rms",
        },
    )
    orders = load_csv(
        accuracy_dir / "observed_orders.csv",
        {
            "method",
            "cfl",
            "N_fine",
            "solution_order_rms",
            "truncation_order_rms",
        },
    )

    made = []
    combined_figure = plot_combined_performance(performance)
    made.extend(
        save_figure(
            combined_figure,
            args.output_dir,
            "fig2-two-blob-performance-combined",
            args.dpi,
            args.overwrite,
        )
    )
    if not args.show:
        plt.close(combined_figure)

    for cfl in args.performance_cfls:
        figure = plot_performance(performance, cfl)
        made.extend(
            save_figure(
                figure,
                args.output_dir,
                f"fig2-two-blob-performance-cfl{cfl_stem(cfl)}",
                args.dpi,
                args.overwrite,
            )
        )
        if not args.show:
            plt.close(figure)

    accuracy_figure = plot_accuracy(accuracy, orders)
    made.extend(
        save_figure(
            accuracy_figure,
            args.output_dir,
            "fig3-two-blob-accuracy",
            args.dpi,
            args.overwrite,
        )
    )
    if args.show:
        plt.show()
    else:
        plt.close(accuracy_figure)

    print(f"performance data: {performance_dir.resolve()}")
    print(f"accuracy data: {accuracy_dir.resolve()}")
    for path in made:
        print(f"wrote {path.resolve()}")


if __name__ == "__main__":
    main()
