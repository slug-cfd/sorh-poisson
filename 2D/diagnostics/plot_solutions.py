import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

PROBLEM_NAMES = {
    "poisson": "Poisson",
    "nonlinear_diffusion": "Nonlinear Diffusion",
    "linear_diffusion": "Linear Diffusion",
}


def _physical_domain_slices(context):
    """Return stored-array slices spanning the configured physical domain."""
    x = np.asarray(context.x)
    y = np.asarray(context.y)
    x_tolerance = 16.0 * np.finfo(float).eps * max(
        1.0, abs(context.xa), abs(context.xb)
    )
    y_tolerance = 16.0 * np.finfo(float).eps * max(
        1.0, abs(context.ya), abs(context.yb)
    )
    x_indices = np.flatnonzero(
        (x >= context.xa - x_tolerance) & (x <= context.xb + x_tolerance)
    )
    y_indices = np.flatnonzero(
        (y >= context.ya - y_tolerance) & (y <= context.yb + y_tolerance)
    )
    if x_indices.size == 0 or y_indices.size == 0:
        raise ValueError(
            "The stored grid does not contain the configured physical domain."
        )
    return (
        slice(int(x_indices[0]), int(x_indices[-1]) + 1),
        slice(int(y_indices[0]), int(y_indices[-1]) + 1),
    )


def plot_solution(U, iter_count, history, context, history_label="Residual"):
    print("Plotting solution...")

    problem_name = PROBLEM_NAMES[context.problem]
    x_slice, y_slice = _physical_domain_slices(context)
    x, y = context.x[x_slice], context.y[y_slice]
    U_plot = U[x_slice, y_slice].T

    nlevels = 15

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1], figure=fig)

    fig.suptitle(f"Results: {problem_name} ({context.method})", fontsize=16)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # contour plot
    cs = ax1.contour(x, y, U_plot, levels=nlevels)
    fig.colorbar(cs, ax=ax1, label=r"$U(x, y)$")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title(r"Contours of $U(x,y)$")

    history_length = min(iter_count, len(history))
    ax2.semilogy(
        range(1, history_length + 1),
        history[:history_length],
        linewidth=2,
    )
    ax2.set_title("Convergence History")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(history_label)
    ax2.grid(True)

    plt.show()
    return fig


def plot_exact_solution(config):
    x_slice, y_slice = _physical_domain_slices(config)
    x, y = config.x[x_slice], config.y[y_slice]

    if getattr(config, "U_exact", None) is not None:
        U = config.U_exact[x_slice, y_slice]
    elif config.phi_exact is not None:
        X, Y = np.meshgrid(x, y, indexing="ij")
        U = config._eval_exact(X, Y)
    else:
        raise ValueError("Cannot plot exact solution: config.phi_exact is not set.")
    U_plot = U.T

    nlevels = 50

    fig = plt.figure(figsize=(8, 6))
    fig.suptitle("Exact Solution", fontsize=16)
    ax = fig.add_subplot(111)

    # contour plot
    cs = ax.contourf(x, y, U_plot, levels=nlevels)
    fig.colorbar(cs, ax=ax, label=r"$U(x, y)$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"Contours of $U(x,y)$")

    plt.show()
    return fig
