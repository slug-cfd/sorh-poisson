import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from problems import poisson2D, nonlinear2D, linear2D
import config2D


problem_map = {
    'poisson': poisson2D,
    'toy_nonlinear': nonlinear2D,
    'toy_linear': linear2D,
}

problem_name_map = {
    'poisson': "Poisson",
    'nonlinear': "Nonlinear",
    'linear': "Linear",
}

method_name_map = {
    'p_unsplit': "Parabolic/Unsplit",
    'p_ADI': "Parabolic/ADI",
    'h_unsplit': "Hyperbolic/Unsplit",
    'h_ADI': "Hyperbolic/ADI",
}


def solve_p2D(config, plot=False):
    if config.problem != "poisson" or config.mg_solver is not None:
        print("invalid problem for poisson solver!")
        return 0

    start = time.perf_counter()

    U, t, iter_count, histories = poisson2D.solve_all(config)

    end = time.perf_counter()
    elapsed = end - start

    ### histories structure ###
    # histories = {
    #     "update": np.array(update_hist),
    #     "residual": np.array(residual_hist),
    #     "exact": np.array(exact_hist),
    #     "best_exact": best_exact,
    #     "best_iter": best_iter,
    # }

    conv_history = histories["exact"]
    res_final = conv_history[-1]

    print("method:", config.method)
    print("iterations:", iter_count)
    print("final residual: ", res_final)
    print(f"best residual: {histories["best_exact"]} | iteration {histories["best_iter"]}")
    # print("time t: ", t)
    print("elapsed time: ", elapsed)


    if plot:
        plot_solution(U, iter_count, conv_history, config)

    return U, t, iter_count, histories


def solve_p2d_multigrid(config, plot=False):
    if config.problem != "poisson" or config.mg_solver is None:
        print("invalid problem for poisson solver!")
        return 0

    U, t, cycle_count, histories = poisson2D.solve_all(config)

    ### histories structure ###
    # histories = {
    #     "update": np.array(update_hist),
    #     "residual": np.array(residual_hist),
    #     "exact": np.array(exact_hist),
    #     "cycle_time": np.array(cycle_time_hist),
    #     "best_exact": best_exact,
    #     "best_iter": best_iter,
    #     "elapsed": elapsed,
    #     "cycles": iter_count,
    #     "U_best": U_best,
    # }

    conv_history = histories["exact"]
    res_final = conv_history[-1]

    print()
    print("method:", config.method)
    print("total time: ", t)
    print("cycle count:", cycle_count)
    print("final residual: ", res_final)

    if plot:
        plot_solution(U, cycle_count, conv_history, config)

    return U, t, cycle_count, histories


def solve_nl2D(config, plot=False):
    if config.problem != "nonlinear":
        print("invalid problem for poisson solver!")
        return 0

    U, t, iter_count, histories = nonlinear2D.solve_all(config)

    ### histories structure ###
    # histories = {
    #     "update": np.array(update_hist),
    #     "U_res_min": np.array(U_res_min),
    #     "best_update": best_update,
    #     "best_iter": best_iter,
    # }

    print("method:", config.method)
    print("iterations:", iter_count)
    print("final residual: ", histories["update_hist"][-1])
    print(f"best residual: {histories["best_update"]} | iteration {histories["best_iter"]}")
    print("time t: ", t)

    if plot:
        plot_solution(U, iter_count, histories["update_hist"], config)

    return U, t, iter_count, histories


def solve_l2D(config, plot=False):
    if config.problem != "linear":
        print("invalid problem for poisson solver!")
        return 0

    U, t, iter_count, res, err_conv = linear2D.solve(config)

    if plot:
        plot_solution(U, iter_count, err_conv, config)

    return U, t, iter_count, res, err_conv


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


def plot_exact_solution(params):
    x, y, rho0, R, G = params.x, params.y, params.rho0, params.R, params.G
    X, Y = np.meshgrid(x, y, indexing='ij')

    U = poisson2D.phi_exact(X, Y, rho0, R, G)
    # rho = poisson2D.f(X, Y, rho0, R)

    U_plot = U.T
    # rho_plot = rho.T

    nlevels = 15

    fig = plt.figure(figsize=(8, 6))

    fig.suptitle(f"Exact Solution", fontsize=16)

    ax = fig.add_subplot(111)

    # contour plot
    cs = ax.contour(x, y, U_plot, levels=nlevels)
    fig.colorbar(cs, ax=ax, label=r"$U(x, y)$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"Contours of $U(x,y)$")

    # plt.savefig("poisson_exact.png", dpi=600)

    plt.show()


def time_compare(config1, config2):
    start_time = time.perf_counter()

    problem_map[config1.problem].solve_all(config1)

    end_time = time.perf_counter()
    time1 = end_time - start_time

    if config1.mg_solver is not None:
        print(f"total time ({config1.method} with multigrid): {time1:.8f} seconds")
    else:
        print(f"total time ({config1.method}): {time1:.8f} seconds")

    start_time = time.perf_counter()

    problem_map[config1.problem].solve_all(config2)

    end_time = time.perf_counter()
    time2 = end_time - start_time

    if config2.mg_solver is not None:
        print(f"total time ({config2.method} with multigrid): {time2:.8f} seconds")
    else:
        print(f"total time ({config2.method}): {time2:.8f} seconds")

    max_time = max(time1, time2)
    min_time = min(time1, time2)

    print(f"speedup: {max_time / min_time:.8f}")
    # print()


solver_map = {
    "poisson_base": solve_p2D,
    "poisson_mg": solve_p2d_multigrid,
    "toy_nonlinear": solve_nl2D,
    "toy_linear": solve_l2D,
}

problem = "poisson"     # test problem
mg_solver = "fmg"       # multigrid solver ("rep_v_cycles", "fmg"; `None` for no multigrid)

solver_name = ""    # do not modify this
if problem == "poisson":
    if mg_solver is not None:
        solver_name = "poisson_mg"
    else:
        solver_name = "poisson_base"
else:
    solver_name = problem

solver = solver_map[solver_name]

if problem == "poisson":
    bc_type = "poisson"
else:
    bc_type = "linear_in_x"

Nx = 64
Ny = 64
cfl = 0.8
iter_max = 4000
norm = "fro"

methods = {
    "poisson_base": ["p_unsplit", "p_ADI", "h_unsplit", "h_ADI"],

    "poisson_mg": ["p_unsplit", "p_ADI", "h_unsplit", "h_ADI"],

    "toy_nonlinear": ["p_unsplit", "h_unsplit", "DS1", "DS2"],

    "toy_linear": [],
}

# indices of array within `methods`
m1 = 0      # config1
m2 = 0      # config2


# h_unsplit (N=32, weighted hybrid): a = b = 0.8: 0.687
# p_unsplit (N=32, non-multigrid): omega 2.294

# p_unsplit multigrid vs. non-multigrid comparison (N=64):
# -> non-multigrid:     omega = 2.390   -> ~1.1s
# -> multigrid (fmg):   omega = 2.300   -> ~0.65s   -> almost 2x speedup!

omegas = {
    "poisson_base": [2.390, 1.81, 0.687, 0.25],

    "poisson_mg": [2.330, 1.0, 0.670, 1.0],

    "toy_nonlinear": [1.55, 0.82, 0.80, 0.80],  # N=32 optimal?

    "toy_linear": [],
}

config1 = config2D.Config(
    problem=problem,
    method=methods[solver_name][m1],
    bc_type=bc_type,
    Nx=Nx,
    Ny=Ny,
    cfl=cfl,
    omega=omegas[solver_name][m1],
    iter_max=iter_max,
    norm=norm,
    f=poisson2D.f,
    phi_exact=poisson2D.phi_exact,
    mg_solver=mg_solver,
)

config2 = config2D.Config(
    problem=problem,
    method=methods[solver_name][m2],
    bc_type=bc_type,
    Nx=Nx,
    Ny=Ny,
    cfl=cfl,
    omega=omegas[solver_name][m2],
    iter_max=iter_max,
    norm=norm,
    f=poisson2D.f,
    phi_exact=poisson2D.phi_exact,
    mg_solver=mg_solver,
)

solver(config1, plot=True)

# example comparison
# print("\n---timed test---")
# time_compare(config, config2)

