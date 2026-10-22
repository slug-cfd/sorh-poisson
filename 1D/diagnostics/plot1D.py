import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from problem_helpers import equation_residual, exact_diagnostic, mixed_mode_components


HISTORY_LABELS = {
    "equation": ("Equation Residual History", "Equation residual norm"),
    "exact": ("Exact Error History", "Exact diagnostic norm"),
    "update": ("Update History", "Update norm"),
}


def _plot_history(axis, iter_count, histories, history_key):
    history_title, history_ylabel = HISTORY_LABELS[history_key]
    history_values = histories[history_key][:iter_count]
    history_iterations = range(1, len(history_values) + 1)
    axis.semilogy(history_iterations, history_values, linewidth=2)
    axis.set_title(history_title)
    axis.set_xlabel("Iteration")
    axis.set_ylabel(history_ylabel)
    axis.grid(True)


def _plot_mixed_mode(U, iter_count, histories, config):
    physical = config.physical_slice
    x = config.x_physical
    components = mixed_mode_components(x)
    exact_error = np.abs(exact_diagnostic(U, config)[physical])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"1D Results: {config.run_display_name}", fontsize=14)

    if config.initial_profile == "mixed":
        for name, values in components.items():
            axes[0, 0].plot(x, values, linewidth=1.5, label=name)
        axes[0, 0].plot(
            x,
            config.U0[physical],
            color="black",
            linewidth=2,
            label="mixed",
        )
        axes[0, 0].legend()
    else:
        axes[0, 0].plot(
            x,
            components[config.initial_profile],
            linewidth=2,
            label=config.initial_profile,
        )
        axes[0, 0].legend()
    axes[0, 0].set_title("Initial Error Profile")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("Error")
    axes[0, 0].grid(True)

    axes[0, 1].plot(
        x,
        config.U0[physical],
        linestyle="--",
        linewidth=1.5,
        label="initial",
    )
    axes[0, 1].plot(x, U[physical], linewidth=2, label="final")
    axes[0, 1].set_title("Initial and Final Error")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("U(x)")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    _plot_history(
        axes[1, 0],
        iter_count,
        histories,
        config.plot_history_norm,
    )

    if config.plot_history_norm == "equation":
        equation_values = equation_residual(U, config)
        spatial_values = np.abs(equation_values[physical])
        spatial_title = "Final Equation Residual"
        spatial_label = "absolute residual"
    else:
        spatial_values = exact_error
        spatial_title = "Final Spatial Error"
        spatial_label = "absolute error"
    axes[1, 1].plot(x, spatial_values, linewidth=2, label=spatial_label)
    axes[1, 1].set_title(spatial_title)
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("Magnitude")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


def plot_solution(U, iter_count, histories, config):
    if config.problem == "linear_mixed_mode":
        _plot_mixed_mode(U, iter_count, histories, config)
        return

    physical = config.physical_slice
    x = config.x_physical
    exact_error = np.abs(exact_diagnostic(U, config)[physical])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"1D Results: {config.run_display_name}", fontsize=14)

    axes[0].plot(x, U[physical], linewidth=2)
    axes[0].set_title("Final Solution")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("U(x)")
    axes[0].grid(True)

    _plot_history(axes[1], iter_count, histories, config.plot_history_norm)

    axes[2].plot(x, exact_error, linewidth=2)
    axes[2].set_title("Exact Diagnostic")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("Residual")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()
