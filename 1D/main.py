import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import jacobi_1D as jacobi
import gauss_seidel_linear as gsl
import sor_linear as sorl

# import jacobi_nonlinear as jnl
import gauss_seidel_nonlinear as gsnl
import sor_nonlinear as sornl
import sor_hyperbolic as sorh
import sorh_linear as sorhl

method_name_map = {
    'jl': "Jacobi (Linear)",
    'gsl': "Gauss Seidel (Linear)",
    'sorl': "SOR (linear)",
    'jnl': "Jacobi (non-linear)",
    'gsnl': "Gauss Seidel (non-linear)",
    'sornl': "SOR (non-linear)",
    'sorh': "SOR (hyperbolic, non-linear)",
    'sorhl': "SOR (hyperbolic, linear)",
}

# switch solver for a single run of a method + set of parameters
def solve(method, N, cfl, iter_max, omega=1.0):
    method_name = method_name_map[method]
    # print(f"Solving - omega={omega}...")

    if method == "jl":
        U, t, iter_count, res, err_conv, err_exact = jacobi.solve(N, cfl, iter_max, linear=True)
    elif method == "gsl":
        U, t, iter_count, res, err_conv, err_exact = gsl.solve(N, cfl, iter_max)
    elif method == "sorl":
        U, t, iter_count, res, err_conv, err_exact = sorl.solve(N, cfl, iter_max, omega)
    elif method == "jnl":
        U, t, iter_count, res, err_conv, err_exact = jacobi.solve(N, cfl, iter_max, linear=False)
    elif method == "gsnl":
        U, t, iter_count, res, err_conv, err_exact = gsnl.solve(N, cfl, iter_max)
    elif method == "sornl":
        U, t, iter_count, res, err_conv, err_exact = sornl.solve(N, cfl, iter_max, omega)
    elif method == "sorh":
        U, t, iter_count, res, err_conv, err_exact = sorh.solve(N, cfl, iter_max, omega)
    elif method == "sorhl":
        U, t, iter_count, res, err_conv, err_exact = sorhl.solve(N, cfl, iter_max, omega)
    else:
        print("invalid method!")
        return 0

    return U, t, iter_count, res, err_conv, err_exact


# plots all the information for one solving run
def plot(method, U, iter_count, err_conv, err_exact):
    method_name = method_name_map[method]

    x = np.linspace(0, 1, N + 1)

    # L2 error against exact solution
    err_exact_norm = np.linalg.norm(err_exact, 2) / np.sqrt(N - 1)
    print("error vs. exact solution (2-norm): ", err_exact_norm)

    # ---- layout: equal-sized axes ----
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 4)
    fig.suptitle(f"Results: {method_name}", fontsize=16)  # ← GLOBAL TITLE

    # top row
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])

    # bottom row (centered, same size)
    ax3 = fig.add_subplot(gs[1, 1:3])

    # ---- plots ----
    ax1.plot(x, U, linewidth=2)
    ax1.set_title(f"Final Solution")
    ax1.set_xlabel("x")
    ax1.set_ylabel("U(x)")
    ax1.grid(True)

    ax2.semilogy(range(iter_count), err_conv[:iter_count], linewidth=2)
    ax2.set_title(f"Convergence History")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Error")
    ax2.grid(True)

    ax3.plot(x, err_exact, linewidth=2)
    ax3.set_title(f"Error (vs. exact solution)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("|x − U(x)|")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


# Ns = np.logspace(3, 7, num=num, base=base)
# omegas = np.arange(0.05, 2.10, 0.05)
# print(omegas)

# iters = np.zeros_like(omegas)

# for i, omega in enumerate(omegas):
#     U, Err, iter_count, res, t = solve(method, N, cfl, iter_max, omega)
#     iters[i] = iter_count
#
# print(iters)
# save as two columns: x  U(x)
# np.savetxt(
#     "solution.dat",
#     np.column_stack((omegas, iters)),
#     header="omega iters",
# )

# plot(method, U, Err, iter_count)


def single_state_test(method, N, cfl, iter_max, omega):
    U, t, iter_count, res, err_conv, err_exact = solve(method, N, cfl, iter_max, omega)

    print("iterations:", iter_count)
    print("final residual: ", res)
    print("time t: ", t)

    plot(method, U, iter_count, err_conv, err_exact)


def generate_map(param_list, test_param='omega', comp_metric='iter_count'):
    method, N, cfl, iter_max = param_list

    omegas = np.arange(0.9, 1.1, 0.002)
    iter_counts = np.zeros_like(omegas)

    for i, omega in enumerate(omegas):
        U, t, iter_count, res, err_conv, err_exact = solve(method, N, cfl, iter_max, omega)
        iter_counts[i] = iter_count

    # save as two columns: x  U(x)
    np.savetxt(
        f"map_{method}_{test_param}_vs_{comp_metric}.dat",
        np.column_stack((omegas, iter_counts)),
        header="omega iter_count",
    )

    print("Done")

def generate_map_adaptive(
    param_list,
    test_param="omega",
    comp_metric="iter_count",
    omega_min=0.01,
    omega_max=1.99,
    d_omega_init=0.01,
    d_omega_min=2e-4,
    d_omega_max=2e-2,
    target_delta_iter=5.0,   # aim for ~this change in iter_count per step
    max_points=5000,
    eps=1e-12,
):
    """
    Adaptive omega sampling:
      - After each new point, estimate |d(iter_count)/d(omega)|
      - Choose next d_omega so that expected change in iter_count ~ target_delta_iter
      - Clamp d_omega to [d_omega_min, d_omega_max]
    """
    method, N, cfl, iter_max = param_list

    omegas = [omega_min]
    iter_counts = []

    # first evaluation
    U, t, iter_count, res, err_conv, err_exact = solve(method, N, cfl, iter_max, omegas[-1])
    iter_counts.append(float(iter_count))

    d_omega = d_omega_init

    while omegas[-1] < omega_max and len(omegas) < max_points:
        omega_next = omegas[-1] + d_omega
        if omega_next > omega_max:
            omega_next = omega_max

        U, t, iter_count, res, err_conv, err_exact = solve(method, N, cfl, iter_max, omega_next)
        omegas.append(omega_next)
        iter_counts.append(float(iter_count))

        # update step size from last two points
        d_iter = abs(iter_counts[-1] - iter_counts[-2])
        d_w = omegas[-1] - omegas[-2]
        slope = d_iter / max(d_w, eps)  # |d iter_count / d omega|

        # pick next step: smaller when slope is large, larger when slope is small
        # expected d_iter ≈ slope * d_omega  => d_omega ≈ target_delta_iter / slope
        d_omega = target_delta_iter / max(slope, eps)

        # clamp to keep progress stable
        d_omega = float(np.clip(d_omega, d_omega_min, d_omega_max))

    omegas = np.array(omegas, dtype=float)
    iter_counts = np.array(iter_counts, dtype=float)

    np.savetxt(
        f"map_{method}_{test_param}_vs_{comp_metric}_n{N}.dat",
        np.column_stack((omegas, iter_counts)),
        header="omega iter_count",
    )

def find_optimal_omega_vs_N(
    method,
    Ns,
    cfl,
    iter_max,
    omega_min=0.55,
    omega_max=0.99,
    d_omega_init=0.02,
    d_omega_min=5e-3,
    d_omega_max=5e-1,
    target_delta_iter=5.0,
    max_points=2000,
    eps=1e-12,
):
    """
    For each N:
      - adaptively sample omega
      - find omega that minimizes iteration count
    """

    results = []

    for N in Ns:
        omegas = [omega_min]
        iters = []

        # first solve
        _, _, iter_count, _, _, _ = solve(method, N, cfl, iter_max, omegas[-1])
        iters.append(float(iter_count))

        omega_opt = omegas[-1]
        iter_min = iters[-1]

        d_omega = d_omega_init

        while omegas[-1] < omega_max and len(omegas) < max_points:
            omega_next = omegas[-1] + d_omega
            if omega_next > omega_max:
                omega_next = omega_max

            _, _, iter_count, _, _, _ = solve(
                method, N, cfl, iter_max, omega_next
            )

            omegas.append(omega_next)
            iters.append(float(iter_count))

            # update minimum
            if iter_count < iter_min:
                iter_min = iter_count
                omega_opt = omega_next

            # adaptive step update
            d_iter = abs(iters[-1] - iters[-2])
            d_w = omegas[-1] - omegas[-2]
            slope = d_iter / max(d_w, eps)

            d_omega = target_delta_iter / max(slope, eps)
            d_omega = float(np.clip(d_omega, d_omega_min, d_omega_max))

        results.append((N, omega_opt, iter_min))

        print(f"N = {N}: Done")

    return np.array(results)


# omega = 2.865
# omega = 0.986       # N = 256 -> ~1600 iterations
# omega = 0.87409     # N = 32 -> 200 iterations
# omega = 0.822735    # N = 16 -> 108 iterations
# omega = 0.65443     # N = 8 -> 55 iterations

# number of iterations for optimal omega values scales linearly with N

method = "sorhl"
cfl = 0.8
iter_max = 10000

N = 32
# omega = 0.65443
# omega = 0.87409
omega = 0.50

single_state_test(method, N, cfl, iter_max, omega)


# omega_min = 0.55
# omega_max = 0.99
# Ns = np.arange(4, 129, 1)

# results = find_optimal_omega_vs_N(
#     method=method,
#     Ns=Ns,
#     cfl=cfl,
#     iter_max=iter_max,
#     omega_min=omega_min,
#     omega_max=omega_max,
# )

# columns: N, omega_opt(N), min_iter(N)
# np.savetxt(
#     f"{method}_optimal_omega_vs_N.dat",
#     results,
#     header="N omega_opt iter_min"
# )


# param_list = [method, N, cfl, iter_max]
# # generate_map(param_list)
# generate_map_adaptive(param_list)
