import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"

PROBLEMS = ("linear", "nonlinear")
SCHEMES = ("parabolic", "hyperbolic")
PLOT_DPI = 400
MARKER_SIZE = 4


def _read_csv(path):
    with Path(path).open() as fh:
        return [_convert_row(row) for row in csv.DictReader(fh)]


def _convert_row(row):
    converted = dict(row)
    for key in ("N", "iter_max", "iter_count", "best_iter"):
        converted[key] = int(float(converted[key]))
    for key in (
        "cfl",
        "omega",
        "linear_v",
        "nonlinear_vmax",
        "hyperbolic_linear_v",
        "tol_update",
        "elapsed_sec",
        "final_update",
        "final_exact",
        "best_exact",
        "pseudo_time",
    ):
        if key in converted and converted[key] != "":
            converted[key] = float(converted[key])
    for key in ("converged", "stable"):
        converted[key] = converted[key] == "True"
    return converted


def _method(problem, scheme):
    return f"sor_{problem}_{scheme}"


def _problem_for_method_name(method):
    return "nonlinear" if "nonlinear" in method else "linear"


def _method_label(method):
    return (
        method.replace("sor_", "SOR ")
        .replace("jacobi_", "Jacobi ")
        .replace("gs_", "GS ")
        .replace("_", " / ")
        .title()
    )


def _sort_key(row):
    return (row["problem"], row["N"], row["scheme"], row["omega"])


def _finite_rows(rows):
    return [
        row for row in rows
        if np.isfinite(row["final_update"])
        and np.isfinite(row["final_exact"])
        and np.isfinite(row["elapsed_sec"])
    ]


def _successful_rows(rows):
    return [
        row for row in _finite_rows(rows)
        if row["converged"] and row["stable"]
    ]


def _best_rows(rows):
    best = {}
    for row in _successful_rows(rows):
        key = (row["problem"], row["method"], row["N"])
        current = best.get(key)
        if current is None or (row["iter_count"], row["elapsed_sec"]) < (
            current["iter_count"],
            current["elapsed_sec"],
        ):
            best[key] = row
    return best


def _range_text(values):
    if not values:
        return "-"
    return f"{min(values):.3g} - {max(values):.3g}"


def _markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _format_float(value, precision=3):
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.{precision}g}"


def baseline_table(baseline_rows):
    rows = []
    for row in sorted(baseline_rows, key=lambda r: (r["problem"], r["N"], r["method"])):
        rows.append(
            [
                row["problem"],
                row["N"],
                _method_label(row["method"]),
                _format_float(row["omega"], 3),
                row["iter_count"],
                _format_float(row["elapsed_sec"], 4),
                row["reason"],
                _format_float(row["final_exact"], 3),
            ]
        )
    return _markdown_table(
        ["Problem", "N", "Method", "Omega", "Iters", "Seconds", "Reason", "Final Exact"],
        rows,
    )


def equal_omega_table(baseline_rows):
    rows = []
    sor_rows = [
        row for row in baseline_rows
        if row["method"].startswith("sor_")
    ]
    grouped = {}
    for row in sor_rows:
        grouped.setdefault((row["problem"], row["N"]), {})[row["scheme"]] = row

    for (problem, N), methods in sorted(grouped.items()):
        par = methods.get("parabolic")
        hyp = methods.get("hyperbolic")
        if par is None or hyp is None:
            continue
        comparable = (
            par["converged"]
            and par["stable"]
            and hyp["converged"]
            and hyp["stable"]
        )
        speedup_iter = (
            par["iter_count"] / hyp["iter_count"]
            if comparable and hyp["iter_count"]
            else None
        )
        speedup_time = (
            par["elapsed_sec"] / hyp["elapsed_sec"]
            if comparable and hyp["elapsed_sec"]
            else None
        )
        rows.append(
            [
                problem,
                N,
                par["iter_count"],
                par["reason"],
                hyp["iter_count"],
                hyp["reason"],
                _format_float(speedup_iter, 3),
                _format_float(par["elapsed_sec"], 4),
                _format_float(hyp["elapsed_sec"], 4),
                _format_float(speedup_time, 3),
            ]
        )
    return _markdown_table(
        [
            "Problem",
            "N",
            "Parabolic Iters",
            "Parabolic Reason",
            "Hyperbolic Iters",
            "Hyperbolic Reason",
            "Iter Ratio P/H",
            "Parabolic Sec",
            "Hyperbolic Sec",
            "Time Ratio P/H",
        ],
        rows,
    )


def baseline_vs_sor_table(baseline_rows, sor_rows):
    best = _best_rows(sor_rows)
    rows = []
    baseline_by_key = {
        (row["problem"], row["method"], row["N"]): row
        for row in _finite_rows(baseline_rows)
    }

    for problem in PROBLEMS:
        Ns = sorted(
            {
                row["N"]
                for row in baseline_rows + sor_rows
                if row["problem"] == problem
            }
        )
        methods = (
            f"jacobi_{problem}_parabolic",
            f"gs_{problem}_parabolic",
            _method(problem, "parabolic"),
            _method(problem, "hyperbolic"),
        )
        for N in Ns:
            for method in methods:
                if method.startswith("sor_"):
                    row = best.get((problem, method, N))
                else:
                    row = baseline_by_key.get((problem, method, N))
                if row is None:
                    rows.append([problem, N, _method_label(method), "-", "-", "-", "-"])
                    continue

                rows.append(
                    [
                        problem,
                        N,
                        _method_label(method),
                        _format_float(row["omega"], 4) if method.startswith("sor_") else "-",
                        row["iter_count"],
                        _format_float(row["elapsed_sec"], 4),
                        row["reason"],
                    ]
                )

    return _markdown_table(
        ["Problem", "N", "Method", "Omega*", "Iters", "Seconds", "Reason"],
        rows,
    )


def optimal_table(sor_rows):
    best = _best_rows(sor_rows)
    rows = []
    for problem in PROBLEMS:
        Ns = sorted({row["N"] for row in sor_rows if row["problem"] == problem})
        for N in Ns:
            par = best.get((problem, _method(problem, "parabolic"), N))
            hyp = best.get((problem, _method(problem, "hyperbolic"), N))
            if par is None or hyp is None:
                continue
            speedup_iter = par["iter_count"] / hyp["iter_count"] if hyp["iter_count"] else np.inf
            speedup_time = par["elapsed_sec"] / hyp["elapsed_sec"] if hyp["elapsed_sec"] else np.inf
            rows.append(
                [
                    problem,
                    N,
                    _format_float(par["omega"], 4),
                    par["iter_count"],
                    _format_float(par["elapsed_sec"], 4),
                    _format_float(hyp["omega"], 4),
                    hyp["iter_count"],
                    _format_float(hyp["elapsed_sec"], 4),
                    _format_float(speedup_iter, 3),
                    _format_float(speedup_time, 3),
                ]
            )
    return _markdown_table(
        [
            "Problem",
            "N",
            "Parabolic Omega*",
            "Parabolic Iters",
            "Parabolic Sec",
            "Hyperbolic Omega*",
            "Hyperbolic Iters",
            "Hyperbolic Sec",
            "Iter Ratio P/H",
            "Time Ratio P/H",
        ],
        rows,
    )


def stability_table(sor_rows):
    rows = []
    for problem in PROBLEMS:
        for N in sorted({row["N"] for row in sor_rows if row["problem"] == problem}):
            for scheme in SCHEMES:
                method = _method(problem, scheme)
                group = [
                    row for row in sor_rows
                    if row["problem"] == problem and row["N"] == N and row["method"] == method
                ]
                stable_omegas = [row["omega"] for row in group if row["stable"]]
                converged_omegas = [
                    row["omega"] for row in group
                    if row["stable"] and row["converged"]
                ]
                unstable_count = sum(not row["stable"] for row in group)
                rows.append(
                    [
                        problem,
                        N,
                        scheme,
                        _range_text(stable_omegas),
                        _range_text(converged_omegas),
                        unstable_count,
                    ]
                )
    return _markdown_table(
        ["Problem", "N", "Scheme", "Stable Omega Range", "Converged Omega Range", "Unstable Runs"],
        rows,
    )


def _plot_omega_metric(sor_rows, problem, N, metric, ylabel, output_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for scheme in SCHEMES:
        method = _method(problem, scheme)
        rows = sorted(
            [
                row for row in sor_rows
                if row["problem"] == problem and row["N"] == N and row["method"] == method
            ],
            key=lambda row: row["omega"],
        )
        converged = _successful_rows(rows)

        if converged:
            ax.plot(
                [row["omega"] for row in converged],
                [row[metric] for row in converged],
                marker="o",
                markersize=MARKER_SIZE,
                linewidth=1.5,
                label=scheme,
            )

    ax.set_title(f"{problem.title()} SOR, N={N}")
    ax.set_xlabel("omega")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=8)
    if metric in ("iter_count", "elapsed_sec"):
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def _plot_optimal_vs_N(sor_rows, output_dir):
    best = _best_rows(sor_rows)
    for problem in PROBLEMS:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for scheme in SCHEMES:
            method = _method(problem, scheme)
            rows = [
                best[(problem, method, N)]
                for N in sorted({row["N"] for row in sor_rows if row["problem"] == problem})
                if (problem, method, N) in best
            ]
            if not rows:
                continue
            Ns = [row["N"] for row in rows]
            axes[0].plot(
                Ns,
                [row["omega"] for row in rows],
                marker="o",
                markersize=MARKER_SIZE,
                label=scheme,
            )
            axes[1].plot(
                Ns,
                [row["iter_count"] for row in rows],
                marker="o",
                markersize=MARKER_SIZE,
                label=scheme,
            )
            axes[2].plot(
                Ns,
                [row["elapsed_sec"] for row in rows],
                marker="o",
                markersize=MARKER_SIZE,
                label=scheme,
            )

        axes[0].set_ylabel("omega*")
        axes[1].set_ylabel("iterations at omega*")
        axes[2].set_ylabel("seconds at omega*")
        for ax in axes:
            ax.set_xlabel("N")
            ax.grid(True, alpha=0.35)
            ax.legend()
        axes[1].set_yscale("log")
        axes[2].set_yscale("log")
        fig.suptitle(f"{problem.title()} optimal SOR scaling")
        fig.tight_layout()
        fig.savefig(output_dir / f"optimal_scaling_{problem}.png", dpi=PLOT_DPI)
        plt.close(fig)


def _plot_stability_maps(sor_rows, output_dir):
    cmap = ListedColormap(["#d73027", "#fee08b", "#1a9850"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    labels = ["unstable", "stable, not converged", "stable and converged"]

    for problem in PROBLEMS:
        for scheme in SCHEMES:
            method = _method(problem, scheme)
            rows = [
                row for row in sor_rows
                if row["problem"] == problem and row["method"] == method
            ]
            if not rows:
                continue

            Ns = sorted({row["N"] for row in rows})
            omegas = sorted({row["omega"] for row in rows})
            omega_index = {omega: idx for idx, omega in enumerate(omegas)}
            N_index = {N: idx for idx, N in enumerate(Ns)}
            status = np.full((len(Ns), len(omegas)), np.nan)

            for row in rows:
                value = 2 if row["stable"] and row["converged"] else 1 if row["stable"] else 0
                status[N_index[row["N"]], omega_index[row["omega"]]] = value

            fig, ax = plt.subplots(figsize=(9, 3.8))
            image = ax.imshow(
                status,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                norm=norm,
            )
            ax.set_title(f"{problem.title()} {scheme} SOR stability")
            ax.set_xlabel("omega")
            ax.set_ylabel("N")
            ax.set_yticks(np.arange(len(Ns)))
            ax.set_yticklabels(Ns)

            max_ticks = 12
            tick_step = max(1, int(np.ceil(len(omegas) / max_ticks)))
            tick_positions = np.arange(0, len(omegas), tick_step)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(
                [f"{omegas[idx]:.3g}" for idx in tick_positions],
                rotation=45,
                ha="right",
            )

            cbar = fig.colorbar(image, ax=ax, ticks=[0, 1, 2])
            cbar.ax.set_yticklabels(labels)
            fig.tight_layout()
            fig.savefig(output_dir / f"stability_map_{problem}_{scheme}.png", dpi=PLOT_DPI)
            plt.close(fig)


def _plot_baseline_vs_sor_N(baseline_rows, sor_rows, output_dir):
    best = _best_rows(sor_rows)
    baseline_methods = (
        "jacobi_linear_parabolic",
        "gs_linear_parabolic",
        "jacobi_nonlinear_parabolic",
        "gs_nonlinear_parabolic",
    )

    for problem in PROBLEMS:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        methods = [
            method for method in baseline_methods
            if _problem_for_method_name(method) == problem
        ]
        for method in methods:
            rows = sorted(
                [
                    row for row in _successful_rows(baseline_rows)
                    if row["problem"] == problem and row["method"] == method
                ],
                key=lambda row: row["N"],
            )
            if not rows:
                continue
            Ns = [row["N"] for row in rows]
            label = _method_label(method)
            axes[0].plot(
                Ns,
                [row["iter_count"] for row in rows],
                marker="o",
                markersize=MARKER_SIZE,
                label=label,
            )
            axes[1].plot(
                Ns,
                [row["elapsed_sec"] for row in rows],
                marker="o",
                markersize=MARKER_SIZE,
                label=label,
            )

        for scheme in SCHEMES:
            method = _method(problem, scheme)
            rows = [
                best[(problem, method, N)]
                for N in sorted({row["N"] for row in sor_rows if row["problem"] == problem})
                if (problem, method, N) in best
            ]
            if not rows:
                continue
            Ns = [row["N"] for row in rows]
            label = f"SOR {scheme} (best omega)"
            axes[0].plot(
                Ns,
                [row["iter_count"] for row in rows],
                marker="o",
                markersize=MARKER_SIZE,
                label=label,
            )
            axes[1].plot(
                Ns,
                [row["elapsed_sec"] for row in rows],
                marker="o",
                markersize=MARKER_SIZE,
                label=label,
            )

        axes[0].set_ylabel("iterations")
        axes[1].set_ylabel("seconds")
        for ax in axes:
            ax.set_xlabel("N")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.35)
            ax.legend(fontsize=7)
        fig.suptitle(f"{problem.title()} baseline vs optimized SOR")
        fig.tight_layout()
        fig.savefig(output_dir / f"baseline_vs_sor_{problem}.png", dpi=PLOT_DPI)
        plt.close(fig)


def make_plots(sor_rows, output_dir, baseline_rows=None):
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for problem in PROBLEMS:
        for N in sorted({row["N"] for row in sor_rows if row["problem"] == problem}):
            _plot_omega_metric(
                sor_rows,
                problem,
                N,
                metric="iter_count",
                ylabel="iteration count",
                output_path=plot_dir / f"omega_vs_iterations_{problem}_n{N}.png",
            )
            _plot_omega_metric(
                sor_rows,
                problem,
                N,
                metric="elapsed_sec",
                ylabel="wall clock seconds",
                output_path=plot_dir / f"omega_vs_time_{problem}_n{N}.png",
            )

    _plot_optimal_vs_N(sor_rows, plot_dir)
    _plot_stability_maps(sor_rows, plot_dir)
    if baseline_rows is not None:
        _plot_baseline_vs_sor_N(baseline_rows, sor_rows, plot_dir)
    return plot_dir


def write_tables(output_dir):
    output_dir = Path(output_dir)
    baseline_rows = _read_csv(output_dir / "baseline_omega1.csv")
    sor_rows = _read_csv(output_dir / "sor_omega_all.csv")
    config_path = output_dir / "experiment_config.json"
    metadata = json.loads(config_path.read_text()) if config_path.exists() else {}
    v_text = _format_float(metadata.get("v"), 3)

    report = [
        "# 1D SOR Performance Summary",
        "",
        f"All generated tests use `v = {v_text}` for `linear_v`, `nonlinear_vmax`, and `hyperbolic_linear_v`.",
        "",
        "## Baseline at omega = 1",
        baseline_table(baseline_rows),
        "",
        "## Parabolic vs Hyperbolic at equal omega = 1",
        "Ratios are only shown when both methods converged stably.",
        equal_omega_table(baseline_rows),
        "",
        "## Baseline vs optimized SOR by N",
        baseline_vs_sor_table(baseline_rows, sor_rows),
        "",
        "## Parabolic vs Hyperbolic at corresponding optimal omega",
        optimal_table(sor_rows),
        "",
        "## Stability and convergence ranges",
        stability_table(sor_rows),
        "",
        "## Generated visual checks",
        "- `plots/stability_map_{problem}_{scheme}.png` shows stability and convergence versus N and omega.",
        "- `plots/baseline_vs_sor_{problem}.png` compares Jacobi/Gauss-Seidel with optimized parabolic and hyperbolic SOR versus N.",
        "- `plots/omega_vs_iterations_{problem}_n{N}.png` and `plots/omega_vs_time_{problem}_n{N}.png` include only stable converged SOR runs.",
        "",
    ]

    report_path = output_dir / "summary_tables.md"
    report_path.write_text("\n".join(report))
    return report_path


def analyze_experiment(output_dir):
    output_dir = Path(output_dir)
    baseline_rows = _read_csv(output_dir / "baseline_omega1.csv")
    sor_rows = _read_csv(output_dir / "sor_omega_all.csv")
    plot_dir = make_plots(sor_rows, output_dir, baseline_rows=baseline_rows)
    report_path = write_tables(output_dir)
    return report_path, plot_dir


def main():
    parser = argparse.ArgumentParser(description="Analyze generated 1D test data.")
    parser.add_argument("--input-dir", default=str(RESULTS_DIR / "sor_performance"))
    args = parser.parse_args()

    report_path, plot_dir = analyze_experiment(args.input_dir)
    print(report_path)
    print(plot_dir)


if __name__ == "__main__":
    main()
