import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import *

def plot_solution(U, iter_count, err_conv, params):
    print(f"Plotting solution...")

    problem_name = problem_name_map[params.problem]

    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
    x, y = params.x[ib:ie+1], params.y[jb:je+1]

    U_interior = U[ib:ie+1, jb:je+1]
    U_plot = U_interior.T

    # fig, ax = plt.subplots()
    # im = ax.imshow(
    #     U_plot, origin="lower", aspect="auto",
    #     extent=[x.min(), x.max(), y.min(), y.max()]
    # )
    # fig.colorbar(im, ax=ax, label="U(x,y)")
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_title("Solution U(x,y)")
    # ax.grid(False)
    #
    # plt.tight_layout()
    # plt.show()

    nlevels = 15

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1], figure=fig)

    fig.suptitle(f"Results: {problem_name} ({params.method})", fontsize=16)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # contour plot
    cs = ax1.contour(x, y, U_plot, levels=nlevels)
    fig.colorbar(cs, ax=ax1, label=r"$U(x, y)$")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title(r"Contours of $U(x,y)$")

    if params.method == "DS1" or params.method == "DS2":
        iter_count -= 1

    ax2.semilogy(range(iter_count), err_conv[:iter_count], linewidth=2)
    ax2.set_title(f"Convergence History")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Error")
    ax2.grid(True)

    # plt.tight_layout()
    # plt.savefig("poisson_parabolic.png", dpi=600)
    plt.show()


def plot_exact_solution(config):
    x, y = config.x, config.y
    X, Y = np.meshgrid(x, y, indexing='ij')

    if getattr(config, "U_exact", None) is not None:
        U = config.U_exact
    elif config.phi_exact is not None:
        U = config._eval_exact(X, Y)
    else:
        raise ValueError("Cannot plot exact solution: config.phi_exact is not set.")
    # rho = poisson2D.f(X, Y, rho0, R)

    U_plot = U.T
    # rho_plot = rho.T

    nlevels = 50

    fig = plt.figure(figsize=(8, 6))
    fig.suptitle(f"Exact Solution", fontsize=16)
    ax = fig.add_subplot(111)

    # contour plot
    cs = ax.contourf(x, y, U_plot, levels=nlevels)
    fig.colorbar(cs, ax=ax, label=r"$U(x, y)$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"Contours of $U(x,y)$")

    # plt.savefig("poisson_exact.png", dpi=600)
    plt.show()
