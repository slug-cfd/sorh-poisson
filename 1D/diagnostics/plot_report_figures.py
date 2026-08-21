import argparse
from dataclasses import dataclass
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_DATA_DIR = BASE_DIR / "results" / "final_report_figure_data"
DEFAULT_FIGURE_DIR = BASE_DIR / "results" / "report_figures"
DEFAULT_FIGURE3_DATA_DIR = DEFAULT_DATA_DIR
DEFAULT_FIGURE4_DATA_DIR = DEFAULT_DATA_DIR

FIGURE5_ENABLED = False

METHOD_ORDER = ["Jacobi", "GS", "SORp", "SORh"]
SOR_ORDER = ["SORh", "SORp"]
SORH_COLOR = "#3D8DFF"
SORP_COLOR = "#E87521"
GS_COLOR = "#4B5563"
JACOBI_COLOR = "#737E8C"
EXACT_COLOR = "#252525"
SECONDARY_COLOR = "#8993A0"
LINEAR_OPTIMUM_COLOR = "#007C78"
NONLINEAR_OPTIMUM_COLOR = "#A64D79"
METHOD_STYLES = {
    "Jacobi": {
        "color": JACOBI_COLOR,
        "marker": "v",
        "linestyle": "--",
        "markerfacecolor": "white",
    },
    "GS": {"color": GS_COLOR, "marker": "D", "linestyle": "-"},
    "SORp": {"color": SORP_COLOR, "marker": "s", "linestyle": "-"},
    "SORh": {
        "color": SORH_COLOR,
        "marker": "o",
        "linestyle": "-",
        "markerfacecolor": "white",
    },
}


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.edgecolor": "#4B5563",
            "axes.linewidth": 0.9,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "grid.color": "#D9DEE5",
            "grid.alpha": 0.75,
            "grid.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


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


def validate_linear_sorh_omega(df, omega_column, source_name):
    group = df[df["method_label"] == "SORh"].copy()
    if group.empty or omega_column not in group:
        raise ValueError(f"{source_name} is missing SORh {omega_column} values.")

    if "N" in group:
        total_intervals = pd.to_numeric(group["N"], errors="coerce")
    else:
        total_intervals = pd.to_numeric(group["interior_unknowns"], errors="coerce") + 1
    observed = pd.to_numeric(group[omega_column], errors="coerce")
    predicted = total_intervals.map(lambda value: 1.0 / (1.0 + math.sin(math.pi / value)))
    offsets = (observed - predicted).abs()
    if offsets.isna().any() or (offsets > 0.05).any():
        bad_rows = group.loc[offsets.isna() | (offsets > 0.05), ["interior_unknowns", omega_column]]
        raise ValueError(
            f"{source_name} contains SORh linear optima inconsistent with the spectral "
            f"prediction:\n{bad_rows.to_string(index=False)}"
        )


def load_figure3_data(data_dir, config):
    linear = load_csv(data_dir / config.data_file)

    combined_columns = {"omega_formula", "omega_linear", "omega_nonlinear"}
    if combined_columns.issubset(linear.columns):
        result = numeric_columns(
            linear,
            ["N", "interior_unknowns", "omega_formula", "omega_linear", "omega_nonlinear"],
        )
        validate_linear_sorh_omega(result, "omega_linear", config.data_file)
        return result

    required_linear = {
        "problem",
        "method_label",
        "N",
        "interior_unknowns",
        "omega_predicted",
        "omega_experimental",
    }
    if not required_linear.issubset(linear.columns):
        missing = sorted(required_linear - set(linear.columns))
        raise ValueError(f"{config.data_file} is missing required columns: {missing}")

    linear = numeric_columns(
        linear,
        ["N", "interior_unknowns", "omega_predicted", "omega_experimental"],
    )
    linear = linear[linear["problem"] == "linear"].copy()
    linear = linear[
        ["method_label", "N", "interior_unknowns", "omega_predicted", "omega_experimental"]
    ].rename(
        columns={
            "omega_predicted": "omega_formula",
            "omega_experimental": "omega_linear",
        }
    )

    nonlinear_path = data_dir / "omega_optima_selected.csv"
    nonlinear = load_csv(nonlinear_path)
    required_nonlinear = {
        "problem",
        "method_label",
        "N",
        "interior_unknowns",
        "omega_experimental",
    }
    if not required_nonlinear.issubset(nonlinear.columns):
        missing = sorted(required_nonlinear - set(nonlinear.columns))
        raise ValueError(f"{nonlinear_path.name} is missing required columns: {missing}")
    nonlinear = numeric_columns(
        nonlinear,
        ["N", "interior_unknowns", "omega_experimental"],
    )
    nonlinear = nonlinear[nonlinear["problem"] == "nonlinear"].copy()
    nonlinear = nonlinear[
        ["method_label", "N", "interior_unknowns", "omega_experimental"]
    ].rename(columns={"omega_experimental": "omega_nonlinear"})

    result = linear.merge(
        nonlinear,
        on=["method_label", "N", "interior_unknowns"],
        how="inner",
        validate="one_to_one",
    )
    expected_rows = len(SOR_ORDER) * 5
    if len(result) != expected_rows:
        raise ValueError(
            f"Canonical Figure 3 merge produced {len(result)} rows; expected {expected_rows}."
        )
    validate_linear_sorh_omega(result, "omega_linear", config.data_file)
    return result


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
            linewidth=2.25,
            markersize=6,
            markeredgewidth=1.2,
            **style,
        )
    configure_log2_x(ax, df["interior_unknowns"].unique(), xlabel="Interior unknowns, N")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major")


def add_nonlinear_scaling_guides(ax, df):
    """Add neutral O(N) and O(N^2) references to Figure 3.2's sweep-count panel."""
    groups = {
        method: df[df["method_label"] == method].sort_values("interior_unknowns")
        for method in SOR_ORDER
    }
    if any(group.empty for group in groups.values()):
        return

    sorh = groups["SORh"]
    sorp = groups["SORp"]
    Ns = sorh["interior_unknowns"].tolist()
    if Ns != sorp["interior_unknowns"].tolist() or len(Ns) < 2:
        return

    # Anchor the guides just below the measured curves so they communicate slope
    # without competing with the method colors or obscuring the data markers.
    reference_N = Ns[1]
    sorh_anchor = 0.78 * float(sorh.iloc[1]["iter_count"])
    sorp_anchor = 0.72 * float(sorp.iloc[1]["iter_count"])
    linear_guide = [sorh_anchor * (N / reference_N) for N in Ns]
    quadratic_guide = [sorp_anchor * (N / reference_N) ** 2 for N in Ns]

    guide_style = {
        "color": "#000000",
        "linewidth": 1.8,
        "zorder": 1,
    }
    ax.plot(Ns, linear_guide, linestyle=":", **guide_style)
    ax.plot(Ns, quadratic_guide, linestyle=":", **guide_style)

    label_style = {
        "color": "#000000",
        "fontsize": 10.5,
        "fontweight": "semibold",
        "ha": "left",
        "va": "center",
        "bbox": {
            "boxstyle": "round,pad=0.16",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 1.0,
        },
        "zorder": 4,
    }
    label_index = max(1, len(Ns) - 2)
    ax.annotate(
        r"$O(N)$ guide",
        xy=(Ns[label_index], linear_guide[label_index]),
        xytext=(10, -9),
        textcoords="offset points",
        **label_style,
    )
    ax.annotate(
        r"$O(N^2)$ guide",
        xy=(Ns[label_index], quadratic_guide[label_index]),
        xytext=(10, -9),
        textcoords="offset points",
        **label_style,
    )


def plot_performance(data_dir, config):
    df = load_csv(data_dir / config.data_file)
    df = numeric_columns(df, ["N", "interior_unknowns", "omega", "iter_count", "elapsed_sec"])
    if config.number == 1:
        validate_linear_sorh_omega(df, "omega", config.data_file)
    gridspec_kw = {"width_ratios": (1.55, 1.0)} if config.number == 2 else None
    fig, axes = plt.subplots(
        1,
        2,
        figsize=config.figsize,
        constrained_layout=True,
        gridspec_kw=gridspec_kw,
    )
    line_by_method(axes[0], df, "iter_count", "Iterations")
    line_by_method(axes[1], df, "elapsed_sec", "Runtime (s)")
    axes[0].set_yscale("log", base=10)
    axes[1].set_yscale("log", base=10)
    if config.number == 2:
        add_nonlinear_scaling_guides(axes[0], df)
    axes[0].set_title("Iterations")
    axes[1].set_title("Runtime")
    axes[1].legend(frameon=False)
    fig.suptitle(config.title)
    return fig


def plot_figure3(data_dir, config):
    df = load_figure3_data(data_dir, config)
    fig, axes = plt.subplots(1, 2, figsize=config.figsize, constrained_layout=True)
    for ax, method in zip(axes, SOR_ORDER, strict=True):
        group = df[df["method_label"] == method].sort_values("N")
        ax.plot(
            group["interior_unknowns"],
            group["omega_formula"],
            label="Linear spectral prediction",
            color=EXACT_COLOR,
            linewidth=2.25,
            linestyle=":",
        )
        ax.plot(
            group["interior_unknowns"],
            group["omega_linear"],
            label="Linear experimental optimum",
            color=LINEAR_OPTIMUM_COLOR,
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.2,
            linewidth=2.25,
            markersize=6,
            linestyle="-",
        )
        ax.plot(
            group["interior_unknowns"],
            group["omega_nonlinear"],
            label="Nonlinear experimental optimum",
            color=NONLINEAR_OPTIMUM_COLOR,
            marker="s",
            markeredgewidth=1.2,
            linewidth=2.25,
            markersize=5.5,
            linestyle="--",
        )
        configure_log2_x(ax, group["interior_unknowns"].unique(), xlabel="Interior unknowns, N")
        ax.set_title(method, color="black")
        ax.set_ylabel(r"$\omega_{\mathrm{opt}}$")
        ax.grid(True, which="major")
    handles = [
        Line2D(
            [0],
            [0],
            color=EXACT_COLOR,
            linestyle=":",
            linewidth=2.25,
            label="Linear spectral prediction",
        ),
        Line2D(
            [0],
            [0],
            color=LINEAR_OPTIMUM_COLOR,
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.2,
            linestyle="-",
            linewidth=2.25,
            markersize=6,
            label="Linear experimental optimum",
        ),
        Line2D(
            [0],
            [0],
            color=NONLINEAR_OPTIMUM_COLOR,
            marker="s",
            markeredgewidth=1.2,
            linestyle="--",
            linewidth=2.25,
            markersize=5.5,
            label="Nonlinear experimental optimum",
        ),
    ]
    labels = [handle.get_label() for handle in handles]
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
            linewidth=2.25,
            color=style["color"],
            linestyle=style["linestyle"],
        )
        exp_omega = group["omega_experimental"].dropna().iloc[0]
        ax.axvline(exp_omega, color=EXACT_COLOR, linewidth=1.5, label="Experimental optimum")
        predicted = group["omega_predicted"].dropna()
        if not predicted.empty:
            ax.axvline(
                predicted.iloc[0],
                color=SECONDARY_COLOR,
                linewidth=1.5,
                linestyle="--",
                label="Predicted optimum",
            )
        cutoff_rows = group[group["is_instability_cutoff"].astype(str) == "True"]
        if not cutoff_rows.empty:
            cutoff = cutoff_rows["omega"].max()
            ax.axvline(
                cutoff,
                color=JACOBI_COLOR,
                linewidth=1.5,
                linestyle=":",
                label="Highest convergent omega",
            )
        ax.set_title(method)
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel("Iterations")
        ax.grid(True, which="major")
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
    paths = [
        output_dir / f"fig{fig_num}-{iteration}.png",
        output_dir / f"fig{fig_num}-{iteration}.pdf",
    ]
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
    fig.savefig(paths[0], dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white")
    return paths


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
    configure_style()
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
            paths = save_figure(fig, fig_num, args.output_dir, args.iteration, args.overwrite)
            made.extend(paths)
        if not args.show:
            plt.close(fig)

    if args.show:
        plt.show()
    for path in made:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
