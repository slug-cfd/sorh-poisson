import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(csv_path):
    df = pd.read_csv(csv_path)

    # enforce numeric columns
    numeric_cols = ["omega", "iter_count", "res_min", "res_max", "res_final", "elapsed_sec"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # stable / convergent may have been written as 0/1
    if "stable" in df.columns:
        df["stable"] = df["stable"].astype(bool)
    if "convergent" in df.columns:
        df["convergent"] = df["convergent"].astype(bool)

    return df


def add_efficiency_metrics(df):
    df = df.copy()

    # useful derived metrics
    df["time_per_iter"] = df["elapsed_sec"] / df["iter_count"].replace(0, np.nan)

    # revised definitions you chose:
    # stable     := res_max <= threshold
    # convergent := stable and res_final <= threshold
    #
    # These should already be in the file, but if you ever want to recompute
    # them externally, you can do that here once you know the threshold.

    return df


def stable_subset(df):
    return df[df["stable"]].copy()


def summarize_all_runs(df):
    print("\n========== OVERALL COUNTS ==========\n")
    summary = (
        df.groupby("method")
        .agg(
            total_runs=("omega", "count"),
            stable_runs=("stable", "sum"),
            convergent_runs=("convergent", "sum"),
            omega_min=("omega", "min"),
            omega_max=("omega", "max"),
        )
        .sort_index()
    )
    print(summary)

    print("\n========== STABLE OMEGA RANGES ==========\n")
    stable_df = stable_subset(df)
    if stable_df.empty:
        print("No stable runs found.")
        return

    stable_ranges = (
        stable_df.groupby("method")
        .agg(
            stable_omega_min=("omega", "min"),
            stable_omega_max=("omega", "max"),
            n_stable=("omega", "count"),
        )
        .sort_index()
    )
    stable_ranges["stable_width"] = (
        stable_ranges["stable_omega_max"] - stable_ranges["stable_omega_min"]
    )
    print(stable_ranges)


def summarize_stable_performance(df):
    stable_df = stable_subset(df)

    print("\n========== STABLE PERFORMANCE SUMMARY ==========\n")
    if stable_df.empty:
        print("No stable runs found.")
        return None

    perf = (
        stable_df.groupby("method")
        .agg(
            n_stable=("omega", "count"),
            omega_min=("omega", "min"),
            omega_max=("omega", "max"),
            iter_mean=("iter_count", "mean"),
            iter_median=("iter_count", "median"),
            iter_min=("iter_count", "min"),
            iter_max=("iter_count", "max"),
            time_mean=("elapsed_sec", "mean"),
            time_median=("elapsed_sec", "median"),
            time_min=("elapsed_sec", "min"),
            time_max=("elapsed_sec", "max"),
            tpi_mean=("time_per_iter", "mean"),
            tpi_median=("time_per_iter", "median"),
        )
        .sort_values("time_mean")
    )
    print(perf)

    print("\n========== BEST STABLE OMEGA BY ITERATIONS ==========\n")
    best_iter_rows = []
    for method, g in stable_df.groupby("method"):
        gbest = g.loc[g["iter_count"].idxmin()]
        best_iter_rows.append(
            {
                "method": method,
                "omega": gbest["omega"],
                "iter_count": gbest["iter_count"],
                "elapsed_sec": gbest["elapsed_sec"],
                "res_final": gbest["res_final"],
            }
        )
    best_iter_df = pd.DataFrame(best_iter_rows).sort_values("iter_count")
    print(best_iter_df.to_string(index=False))

    print("\n========== BEST STABLE OMEGA BY TIME ==========\n")
    best_time_rows = []
    for method, g in stable_df.groupby("method"):
        gbest = g.loc[g["elapsed_sec"].idxmin()]
        best_time_rows.append(
            {
                "method": method,
                "omega": gbest["omega"],
                "elapsed_sec": gbest["elapsed_sec"],
                "iter_count": gbest["iter_count"],
                "res_final": gbest["res_final"],
            }
        )
    best_time_df = pd.DataFrame(best_time_rows).sort_values("elapsed_sec")
    print(best_time_df.to_string(index=False))

    return {
        "perf": perf,
        "best_iter": best_iter_df,
        "best_time": best_time_df,
    }


def plot_iter_vs_omega(df, outdir):
    stable_df = stable_subset(df)
    if stable_df.empty:
        return

    plt.figure(figsize=(8, 5))
    for method, g in stable_df.groupby("method"):
        g = g.sort_values("omega")
        plt.plot(g["omega"], g["iter_count"], marker="o", label=method)

    plt.xlabel(r"$\omega$")
    plt.ylabel("Iteration count")
    plt.title("Stable runs: iteration count vs omega")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "stable_iter_vs_omega.png", dpi=200)
    plt.close()


def plot_time_vs_omega(df, outdir):
    stable_df = stable_subset(df)
    if stable_df.empty:
        return

    plt.figure(figsize=(8, 5))
    for method, g in stable_df.groupby("method"):
        g = g.sort_values("omega")
        plt.plot(g["omega"], g["elapsed_sec"], marker="o", label=method)

    plt.xlabel(r"$\omega$")
    plt.ylabel("Elapsed time [s]")
    plt.title("Stable runs: elapsed time vs omega")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "stable_time_vs_omega.png", dpi=200)
    plt.close()


def plot_time_per_iter_vs_omega(df, outdir):
    stable_df = stable_subset(df)
    if stable_df.empty:
        return

    plt.figure(figsize=(8, 5))
    for method, g in stable_df.groupby("method"):
        g = g.sort_values("omega")
        plt.plot(g["omega"], g["time_per_iter"], marker="o", label=method)

    plt.xlabel(r"$\omega$")
    plt.ylabel("Time per iteration [s]")
    plt.title("Stable runs: time per iteration vs omega")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "stable_time_per_iter_vs_omega.png", dpi=200)
    plt.close()


def plot_time_vs_iters(df, outdir):
    stable_df = stable_subset(df)
    if stable_df.empty:
        return

    plt.figure(figsize=(8, 5))
    for method, g in stable_df.groupby("method"):
        plt.scatter(g["iter_count"], g["elapsed_sec"], label=method)

    plt.xlabel("Iteration count")
    plt.ylabel("Elapsed time [s]")
    plt.title("Stable runs: elapsed time vs iteration count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "stable_time_vs_iter.png", dpi=200)
    plt.close()


def plot_stable_ranges(df, outdir):
    stable_df = stable_subset(df)
    if stable_df.empty:
        return

    methods = sorted(stable_df["method"].unique())
    y = np.arange(len(methods))

    omega_lo = []
    omega_hi = []

    for method in methods:
        g = stable_df[stable_df["method"] == method]
        omega_lo.append(g["omega"].min())
        omega_hi.append(g["omega"].max())

    plt.figure(figsize=(8, 4.5))
    for i, method in enumerate(methods):
        plt.hlines(y=i, xmin=omega_lo[i], xmax=omega_hi[i], linewidth=3)
        plt.plot([omega_lo[i], omega_hi[i]], [i, i], "o")

    plt.yticks(y, methods)
    plt.xlabel(r"$\omega$")
    plt.ylabel("Method")
    plt.title("Stable omega ranges by method")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "stable_omega_ranges.png", dpi=200)
    plt.close()


def plot_best_stable_bars(summary_dict, outdir):
    if summary_dict is None:
        return

    best_iter = summary_dict["best_iter"].copy()
    best_time = summary_dict["best_time"].copy()

    plt.figure(figsize=(8, 5))
    plt.bar(best_iter["method"], best_iter["iter_count"])
    plt.ylabel("Minimum stable iteration count")
    plt.title("Best stable omega by iteration count")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(outdir / "best_stable_itercount_bar.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(best_time["method"], best_time["elapsed_sec"])
    plt.ylabel("Minimum stable elapsed time [s]")
    plt.title("Best stable omega by elapsed time")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(outdir / "best_stable_time_bar.png", dpi=200)
    plt.close()


def save_summary_tables(summary_dict, outdir):
    if summary_dict is None:
        return

    summary_dict["perf"].to_csv(outdir / "stable_performance_summary.csv")
    summary_dict["best_iter"].to_csv(outdir / "best_stable_by_iter.csv", index=False)
    summary_dict["best_time"].to_csv(outdir / "best_stable_by_time.csv", index=False)


def analyze_omega_sweep(csv_path, outdir=None):
    csv_path = Path(csv_path)

    if outdir is None:
        outdir = csv_path.parent / f"{csv_path.stem}_analysis"
    else:
        outdir = Path(outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    df = load_results(csv_path)
    df = add_efficiency_metrics(df)

    summarize_all_runs(df)
    summary_dict = summarize_stable_performance(df)

    plot_iter_vs_omega(df, outdir)
    plot_time_vs_omega(df, outdir)
    plot_time_per_iter_vs_omega(df, outdir)
    plot_time_vs_iters(df, outdir)
    plot_stable_ranges(df, outdir)
    plot_best_stable_bars(summary_dict, outdir)
    save_summary_tables(summary_dict, outdir)

    print(f"\nSaved analysis outputs to: {outdir}\n")

    return df, summary_dict


if __name__ == "__main__":
    # change this to your actual file
    csv_file = "results/omega_Nx32_Ny32.csv"
    analyze_omega_sweep(csv_file)
