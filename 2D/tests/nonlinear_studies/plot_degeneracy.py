"""Plot degeneracy speedup from a completed degeneracy study."""

import argparse
import csv
from pathlib import Path

from .common import write_csv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = (
        args.results / "best_by_case.csv"
        if args.results.is_dir()
        else args.results
    )
    output_dir = args.output_dir or source.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    grouped = {}
    for row in rows:
        grouped.setdefault(row["coefficient_label"], {})[row["method"]] = row

    summary = []
    for label, methods in grouped.items():
        if not {"h_unsplit", "p_unsplit"}.issubset(methods):
            continue
        h_row = methods["h_unsplit"]
        p_row = methods["p_unsplit"]
        summary.append({
            "coefficient_label": label,
            "ellipticity_ratio": float(h_row["ellipticity_ratio"]),
            "sorh_sweeps": int(h_row["sweeps"]),
            "sorp_sweeps": int(p_row["sweeps"]),
            "sweep_speedup_sorh": int(p_row["sweeps"]) / int(h_row["sweeps"]),
            "sorh_elapsed_sec": float(h_row["elapsed_sec"]),
            "sorp_elapsed_sec": float(p_row["elapsed_sec"]),
            "elapsed_speedup_sorh": (
                float(p_row["elapsed_sec"]) / float(h_row["elapsed_sec"])
            ),
        })
    summary.sort(key=lambda row: row["ellipticity_ratio"])
    write_csv(output_dir / "degeneracy_speedup.csv", summary)

    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    x = [row["ellipticity_ratio"] for row in summary]
    y = [row["sweep_speedup_sorh"] for row in summary]
    axis.plot(x, y, marker="o")
    for row in summary:
        axis.annotate(
            row["coefficient_label"],
            (row["ellipticity_ratio"], row["sweep_speedup_sorh"]),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.axhline(1.0, color="0.5", linewidth=1.0, linestyle="--")
    axis.set_xlabel(r"Ellipticity ratio $v_{\min}/v_{\max}$")
    axis.set_ylabel("Sweep speedup (SORp / SORh)")
    axis.set_title("Effect of nonlinear elliptic degeneracy")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "degeneracy_speedup.png", dpi=180)
    plt.close(figure)
    print(f"wrote {output_dir / 'degeneracy_speedup.csv'}")
    print(f"wrote {output_dir / 'degeneracy_speedup.png'}")


if __name__ == "__main__":
    main()
