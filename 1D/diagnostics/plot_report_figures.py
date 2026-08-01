import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_DATA_DIR = (
    BASE_DIR
    / "results"
    / "final_report_figure_data"
    / "consistency_rerun_20260801"
)
DEFAULT_FIGURE_DIR = BASE_DIR / "results" / "report_figures"
DEFAULT_FIGURE3_DATA_DIR = DEFAULT_DATA_DIR
DEFAULT_FIGURE4_DATA_DIR = DEFAULT_DATA_DIR

FIGURE5_ENABLED = False

METHOD_ORDER = ["Jacobi", "GS", "SORp", "SORh"]
SOR_ORDER = ["SORh", "SORp"]
METHOD_STYLES = {
    "Jacobi": {"color": "#3b82f6", "marker": "o", "linestyle": "-"},
    "GS": {"color": "#10b981", "marker": "s", "linestyle": "-"},
    "SORp": {"color": "#f59e0b", "marker": "^", "linestyle": "-"},
    "SORh": {"color": "#ef4444", "marker": "D", "linestyle": "-"},
}
PREDICTED_STYLE = {"color": "#475569", "marker": "o", "linestyle": "--"}
EXPERIMENTAL_STYLE = {"color": "#ef4444", "marker": "D", "linestyle": "-"}
NONLINEAR_STYLE = {"color": "#10b981", "marker": "s", "linestyle": "-."}


@dataclass
class FigureConfig:
    number: int
    data_file: str
    figsize: tuple[float, float]
    title: str


FIGURE_CONFIGS = {
    1: FigureConfig(
        number=1,
        data_file="figure1_linear_performance.csv",
        figsize=(8.8, 3.6),
        title="Linear Problem Performance",
    ),
    2: FigureConfig(
        number=2,
        data_file="figure2_nonlinear_performance.csv",
        figsize=(8.8, 3.6),
        title="Nonlinear Problem Performance",
    ),
    3: FigureConfig(
        number=3,
        data_file="figure3_omega_optima.csv",
        figsize=(8.8, 3.6),
        title="Optimal Relaxation Parameter",
    ),
    4: FigureConfig(
        number=4,
        data_file="figure4_sensitivity_converged.csv",
        figsize=(8.8, 3.6),
        title="Sensitivity to Relaxation Parameter",
    ),
    5: FigureConfig(
        number=5,
        data_file="figure5_nonlinear_final_error.csv",
        figsize=(5.2, 3.8),
        title="Final Numerical Error",
    ),
}


def parse_figures(values):
    if not values:
        return [1, 2, 3, 4]
    result = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip().lower()
            if not part:
                continue
            if part == "all":
                result.extend([1, 2, 3, 4])
            else:
                if part.startswith("figure"):
                    part = part.replace("figure", "", 1)
                result.append(int(part))
    ordered = []
    for fig_num in result:
        if fig_num not in FIGURE_CONFIGS:
            raise ValueError(f"Unknown figure number: {fig_num}")
        if fig_num == 5 and not FIGURE5_ENABLED:
            raise RuntimeError(
                "Figure 5 plotting is intentionally disabled because the data is incomplete. "
                "Edit FIGURE5_ENABLED in this script after the data are ready."
            )
        if fig_num not in ordered:
            ordered.append(fig_num)
    return ordered


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    return pd.read_csv(path)


def numeric_columns(df, columns):
    for column in columns:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def configure_log2_x(ax, Ns, xlabel="Interior unknowns"):
    ticks = sorted(int(value) for value in Ns)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(value) for value in ticks])
    ax.set_xlabel(xlabel)


def line_by_method(ax, df, y_col, ylabel):
    for method in METHOD_ORDER:
        group = df[df["method_label"] == method].sort_values("interior_unknowns")
        if group.empty:
            continue
        style = METHOD_STYLES[method]
        ax.plot(
            group["interior_unknowns"],
            group[y_col],
            label=method,
            linewidth=1.8,
            markersize=4,
            **style,
        )
    configure_log2_x(ax, df["interior_unknowns"].unique(), xlabel="Interior Unknowns (N)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_performance(data_dir, config):
    df = load_csv(data_dir / config.data_file)
    df = numeric_columns(df, ["N", "interior_unknowns", "iter_count", "elapsed_sec"])
    fig, axes = plt.subplots(1, 2, figsize=config.figsize, constrained_layout=True)
    line_by_method(axes[0], df, "iter_count", "Iterations")
    line_by_method(axes[1], df, "elapsed_sec", "Runtime (s)")
    axes[0].set_yscale("log", base=10)
    axes[1].set_yscale("log", base=10)
    axes[0].set_title("Iterations")
    axes[1].set_title("Runtime")
    axes[1].legend(frameon=False)
    fig.suptitle(config.title)
    return fig


def plot_figure3(data_dir, config):
    df = load_csv(data_dir / config.data_file)
    df = numeric_columns(
        df,
        [
            "N",
            "interior_unknowns",
            "omega_formula",
            "omega_linear",
            "omega_nonlinear",
        ],
    )
    fig, axes = plt.subplots(1, 2, figsize=config.figsize, constrained_layout=True)
    for ax, method in zip(axes, SOR_ORDER, strict=True):
        group = df[df["method_label"] == method].sort_values("N")
        ax.plot(
            group["interior_unknowns"],
            group["omega_formula"],
            label="Linear formula",
            linewidth=1.8,
            markersize=4,
            **PREDICTED_STYLE,
        )
        ax.plot(
            group["interior_unknowns"],
            group["omega_linear"],
            label="Linear experimental optimum",
            linewidth=1.8,
            markersize=4,
            **EXPERIMENTAL_STYLE,
        )
        ax.plot(
            group["interior_unknowns"],
            group["omega_nonlinear"],
            label="Nonlinear experimental optimum",
            linewidth=1.8,
            markersize=4,
            **NONLINEAR_STYLE,
        )
        configure_log2_x(ax, group["interior_unknowns"].unique(), xlabel="Interior Unknowns (M)")
        ax.set_title(method)
        ax.set_ylabel(r"$\omega_{\mathrm{opt}}$")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(config.title)
    return fig


def plot_figure4(data_dir, config):
    df = load_csv(data_dir / config.data_file)
    df = numeric_columns(
        df,
        ["omega", "omega_experimental", "omega_predicted", "iter_count"],
    )
    fig, axes = plt.subplots(1, 2, figsize=config.figsize, constrained_layout=True)
    for ax, method in zip(axes, SOR_ORDER, strict=True):
        group = df[df["method_label"] == method].sort_values("omega")
        style = METHOD_STYLES[method]
        ax.plot(
            group["omega"],
            group["iter_count"],
            label=method,
            linewidth=1.8,
            color=style["color"],
        )
        exp_omega = group["omega_experimental"].dropna().iloc[0]
        ax.axvline(exp_omega, color="#111827", linewidth=1.2, label="Experimental optimum")
        predicted = group["omega_predicted"].dropna()
        if not predicted.empty:
            ax.axvline(
                predicted.iloc[0],
                color="#64748b",
                linewidth=1.2,
                linestyle="--",
                label="Predicted optimum",
            )
        cutoff_rows = group[group["is_instability_cutoff"].astype(str) == "True"]
        if not cutoff_rows.empty:
            cutoff = cutoff_rows["omega"].max()
            ax.axvline(
                cutoff,
                color="#6b7280",
                linewidth=1.2,
                linestyle=":",
                label="Highest convergent omega",
            )
        ax.set_title(method)
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel("Iterations")
        ax.grid(True, alpha=0.3)
    handles = []
    labels = []
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels, strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=min(4, len(labels)),
        frameon=False,
    )
    fig.suptitle(config.title)
    return fig


def plot_figure5(data_dir, config):
    df = load_csv(data_dir / config.data_file)
    df = df[
        (df["converged"].astype(str) == "True")
        & (df["stable"].astype(str) == "True")
        & (df["reason"] == "tol_update_and_error_plateau")
    ].copy()
    df = numeric_columns(df, ["N", "interior_unknowns", "plateau_error_mean"])
    fig, ax = plt.subplots(1, 1, figsize=config.figsize, constrained_layout=True)
    for method in METHOD_ORDER:
        group = df[df["method_label"] == method].sort_values("interior_unknowns")
        if group.empty:
            continue
        ax.plot(
            group["interior_unknowns"],
            group["plateau_error_mean"],
            label=method,
            linewidth=1.8,
            markersize=4,
            **METHOD_STYLES[method],
        )
    if not df.empty:
        Ns = sorted(df["interior_unknowns"].unique())
        ref_N = Ns[0]
        ref_error = df[df["interior_unknowns"] == ref_N]["plateau_error_mean"].median()
        reference = [ref_error * (ref_N / N) ** 2 for N in Ns]
        ax.plot(Ns, reference, color="#111827", linestyle="--", linewidth=1.2, label="2nd order")
        configure_log2_x(ax, Ns)
    ax.set_yscale("log", base=10)
    ax.set_ylabel(r"$\mathrm{NumErr}^{(\mathrm{final})}$")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(frameon=False)
    fig.suptitle(config.title)
    return fig


def save_figure(fig, fig_num, output_dir, iteration, overwrite):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"fig{fig_num}-{iteration}.png"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Create final-report figures from collected 1D data.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--figure3-data-dir", type=Path, default=DEFAULT_FIGURE3_DATA_DIR)
    parser.add_argument("--figure4-data-dir", type=Path, default=DEFAULT_FIGURE4_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--figures", nargs="+", default=["1", "2", "3", "4"])
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    figures = parse_figures(args.figures)
    made = []
    for fig_num in figures:
        config = FIGURE_CONFIGS[fig_num]
        if fig_num in (1, 2):
            fig = plot_performance(args.data_dir, config)
        elif fig_num == 3:
            fig = plot_figure3(args.figure3_data_dir, config)
        elif fig_num == 4:
            fig = plot_figure4(args.figure4_data_dir, config)
        elif fig_num == 5:
            fig = plot_figure5(args.data_dir, config)
        else:
            raise AssertionError(f"Unhandled figure {fig_num}")

        if args.save:
            path = save_figure(fig, fig_num, args.output_dir, args.iteration, args.overwrite)
            made.append(path)
        if not args.show:
            plt.close(fig)

    if args.show:
        plt.show()
    for path in made:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
