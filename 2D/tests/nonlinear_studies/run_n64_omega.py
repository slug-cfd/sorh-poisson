"""Run selected N=64 SORp omega cases without repeating the full calibration."""

import argparse
from pathlib import Path

from .common import (
    mark_search_summary,
    run_omega_candidates,
    save_study,
    select_first_converged_branch,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--omega",
        type=float,
        nargs="+",
        required=True,
        help="one or more SORp omega values",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    omegas = sorted(set(args.omega))
    rows = run_omega_candidates(
        "p_unsplit",
        64,
        omegas,
        stage="omega_n64_manual",
    )

    successful = [row for row in rows if row["converged"] and row["stable"]]
    summary = []
    if successful:
        best = select_first_converged_branch(rows)
        summary.append(mark_search_summary(best, omegas))

    output = save_study(
        "omega_n64_manual",
        rows,
        summary,
        {
            "description": "Manual N=64 SORp omega tests",
            "ordering": "lexicographic",
            "cfl": 0.8,
        },
        args.output_dir,
    )

    for row in rows:
        print(
            f"omega={row['omega']:.6g} converged={row['converged']} "
            f"sweeps={row['sweeps']} reason={row['reason']} "
            f"residual={row['residual_final']:.6e}"
        )
    print(f"results: {output}")


if __name__ == "__main__":
    main()
