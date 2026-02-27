import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from problems import poisson2D, nonlinear2D, linear2D
import config2D

problem_name_map = {
    'poisson': "Poisson",
    'nonlinear': "Nonlinear",
    'linear': "Linear",
}

# all 2D methods
# method_name_map = {
#     'parabolic': "Poisson (parabolic)",
#     'unsplit': "Poisson (unsplit)",
#     'DS1': "Poisson (DS1)",
#     'DS2': "Poisson (DS2)"
# }

def solve(params):
    problem = params.problem

    if problem == "poisson":
        U, t, iter_count, res, err_conv = poisson2D.solve(params)
    elif problem == "nonlinear":
        U, t, iter_count, res, err_conv = nonlinear2D.solve(params)
    elif problem == "linear":
        U, t, iter_count, res, err_conv = linear2D.solve(params)
    else:
        print("invalid problem!")
        return 0

    return U, t, iter_count, res, err_conv


def plot(U, iter_count, err_conv, params):
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


def time_compare(params1, params2):
    start_time = time.perf_counter()

    solve(params1)

    end_time = time.perf_counter()
    time1 = end_time - start_time

    print(f"total time ({params1.method}): {time1:.8f} seconds")

    start_time = time.perf_counter()

    solve(params2)

    end_time = time.perf_counter()
    time2 = end_time - start_time

    print(f"total time ({params2.method}): {time2:.8f} seconds")

    max_time = max(time1, time2)
    min_time = min(time1, time2)

    print(f"speedup: {max_time / min_time:.8f}")
    print()


### Notes ###
# convergence is weird for sornl2 (for nonlinear BCs)
# they all seemingly need ~200 iterations to converge
# but the residual value stays at a fixed value (~0.1)
# meaning the solution likely fluctuates around two fixed points
# iter_max = 200

# iteration counts for Nx=Ny=16, sinusoidal BCs (x2?)
#   jl2                 -> 272 iterations
#   gsl2 (with r/b gs)  -> 173 iterations
#   sorl2 (omega=1.50)  -> 118 iterations

# -> similar results for linear solution profile (U(x, y) = x)

# all BCs that I tested for SOR 2D (linear and nonlinear)
# seem to have minimum iterations around omega = 1.5~1.55
# and become unstable around omega = 1.6

# omega = 1.60    # sornl2 optimal -> 396 iterations
# omega = 0.80    # sorh2 optimal  -> 78 iterations
# omega = 0.84    # sorh2u optimal -> 123 iterations

# sorh2 best: 78 iterations (omega=0.8, DS1)
# DS1 and DS2 perform nearly identically
# DS1/2: significantly faster, slightly smaller omega

# unsplit method (sorh2u) still performs much better than
# parabolic version (sornl2), but slightly worse than the
# split version (sorh2)

# HOWEVER: while the solution structures between sornl2
# and sorh2 look slightly different, the solutions for
# sornl2 and sorh2u look absolutely identical.


# method = "sornl2"
# omega = 1.60

# N = 16
# method = "sorh2"
# omega = 0.80

# N = 32 -> crashes at cfl ~ 1.323
# method = "sorh2"
# omega = 0.86

# N = 16
# method = "sorh2_unsplit"
# omega = 0.84

# N = 32 -> crashes at cfl ~ 1.23
# method = "sorh2_unsplit"
# omega = 0.89

# for N = 32, both split and unsplit versions start showing signs of instability
# at similar cfl values, both at cfl~1.2. unsplit version becomes unstable
# slightly sooner, with the solution structure for the split version staying
# mostly intact until cfl~1.323, but still, both are unstable after cfl~1.2.

# N = 64 case is very similar for both split and unsplit
# unstable cfl value does not change significantly for either method

# it seems like for increasing N, the point at which the methods are unstable
# actually doesn't change much. for N = 16, 32, 64, the stability threshold is
# somewhere around cfl~1.2-1.4.

# omega = 0.0741  # N=32 -> 366 iterations, res = 0.4493928629873612
# omega = 0.0196  # N=64 -> 3265 iterations, res = 0.44457492162560386

# omega = 0.1039  # N=32, dt/=4 -> 163 iterations, res = 0.44940645391034956

# !! parabolic solver with omega=2.30 gets 67 iterations!
#    -> also seemingly stable for all N, which hyperbolic solver is not!
# !!!! parabolic w/ omega=2.290 and modified dt gets 37 iterations with very low residual!!
# omega = 2.309     # N = 128


problem = "poisson"

if problem == "poisson":
    bc_type = "poisson"
else:
    bc_type = "linear_in_x"

# method = "parabolic_jacobi"
# omega = 0.8

Nx = 32
Ny = 32
cfl = 0.8
iter_max = 4000

methods = ["parabolic", "unsplit", "DS1", "DS2"]

# Poisson parameters (N=32)
omegas_p = [2.292, 0.919, 0.68, 0.63]

# N = 32 (optimal omega values)
# -> parabolic: 2.292 -> 71 iterations
# -> hyperbolic: 0.919 -> 93 iterations

# N=64: 0.969

# Nonlinear parameters (N=32)
omegas_n = [1.55, 0.82, 0.80, 0.80]

# use this to set the method (index of "methods" array)
m = 0

norm = "fro"

params = config2D.Config(
    problem=problem,
    method=methods[m],
    bc_type=bc_type,
    Nx=Nx,
    Ny=Ny,
    cfl=cfl,
    omega=omegas_p[m],
    iter_max=iter_max,
    norm=norm
)

U, t, iter_count, res, err_conv = solve(params)

print("method:", params.method)
print("iterations:", iter_count)
print("final residual: ", res)
print("time t: ", t)

plot(U, iter_count, err_conv, params)

# plot phi_exact for Poisson test eqn
plot_exact_solution(params)

params2 = config2D.Config(
    problem=problem,
    method=methods[1],
    bc_type=bc_type,
    Nx=Nx,
    Ny=Ny,
    cfl=cfl,
    omega=omegas_p[1],
    iter_max=iter_max,
    norm=norm
)

# example comparison
# print("\n---timed test---")
# time_compare(params, params2)

