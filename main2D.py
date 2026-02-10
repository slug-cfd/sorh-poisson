import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import jacobi_linear_2D as jl2
import gauss_seidel_linear_2D as gsl2
import sor_linear_2D as sorl2
import sor_nonlinear_2D as sornl2
import sor_hyperbolic_2D as sorh2
import sorh_unsplit_2D as sorh2u

import time

# all 2D methods
method_name_map = {
    'jl2': "Jacobi (Linear)",
    'gsl2': "Gauss Seidel (Linear)",
    'sorl2': "SOR (linear)",
    'sornl2': "SOR (parabolic, non-linear)",
    'sorh2': "SOR (hyperbolic, non-linear, split)",
    'sorh2_unsplit': "SOR (hyperbolic, nonlinear, unsplit)",
}

def solve(method, Nx, Ny, cfl, iter_max, omega):
    print(f"Solving ({method})...")

    if method == "jl2":
        U, t, iter_count, res, err_conv = jl2.solve(Nx, Ny, cfl, iter_max)
    elif method == "gsl2":
        U, t, iter_count, res, err_conv = gsl2.solve(Nx, Ny, cfl, iter_max)
    elif method == "sorl2":
        U, t, iter_count, res, err_conv = sorl2.solve(Nx, Ny, cfl, iter_max, omega)
    elif method == "sornl2":
        U, t, iter_count, res, err_conv = sornl2.solve(Nx, Ny, cfl, iter_max, omega)
    elif method == "sorh2":
        U, t, iter_count, res, err_conv = sorh2.solve(Nx, Ny, cfl, iter_max, omega)
    elif method == "sorh2_unsplit":
        U, t, iter_count, res, err_conv = sorh2u.solve(Nx, Ny, cfl, iter_max, omega)
    else:
        print("invalid method!")
        return 0

    return U, t, iter_count, res, err_conv


def plot(U, Nx, Ny, err_conv):
    print(f"Plotting solution...")

    method_name = method_name_map[method]

    x = np.linspace(0, 1, Nx+1)
    y = np.linspace(0, 1, Ny+1)

    U_plot = U.T

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

    nlevels = (Nx + Ny)

    # ---- layout: equal-sized axes ----
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 13)
    fig.suptitle(f"Results: {method_name}", fontsize=16)  # ← GLOBAL TITLE

    # top row
    ax1 = fig.add_subplot(gs[0, 0:7])
    ax2 = fig.add_subplot(gs[0, 7:13])

    # contour plot
    cs = ax1.contourf(x, y, U_plot, levels=nlevels)
    fig.colorbar(cs, ax=ax1, label="U(x,y)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Contours of U(x,y)")

    ax2.semilogy(range(iter_count), err_conv[:iter_count], linewidth=2)
    ax2.set_title(f"Convergence History")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Error")
    ax2.grid(True)

    plt.tight_layout()
    # plt.savefig("sornl2_unsplit_bc0.png", dpi=600)
    plt.show()


def time_compare(Nx, Ny, cfl, iter_max):
    method = "sorh2"
    omega = 0.80

    start_time = time.perf_counter()

    solve(method, Nx, Ny, cfl, iter_max, omega)

    end_time = time.perf_counter()
    sorh2_time = end_time - start_time

    print(f"total time (sorh2): {sorh2_time:.8f} seconds")

    method = "sornl2"
    omega = 1.60

    start_time = time.perf_counter()

    solve(method, Nx, Ny, cfl, iter_max, omega)

    end_time = time.perf_counter()
    sornl2_time = end_time - start_time

    print(f"total time (sornl2): {sornl2_time:.8f} seconds")

    method = "sorh2_unsplit"
    omega = 0.84

    start_time = time.perf_counter()

    solve(method, Nx, Ny, cfl, iter_max, omega)

    end_time = time.perf_counter()
    sorh2u_time = end_time - start_time

    print(f"total time (sorh2u): {sorh2u_time:.8f} seconds")

    print()
    print(f"speedup (sorh2 vs. sornl2): {sornl2_time / sorh2_time:.8f}")
    print(f"speedup (sorh2u vs. sornl2): {sornl2_time / sorh2u_time:.8f}")
    print(f"speedup (sorh2 vs. sorh2u): {sorh2u_time / sorh2_time:.8f}")



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

# unsplit method (sorh2u) still performs much better than
# parabolic version (sornl2), but slightly worse than the
# split version (sorh2)

# HOWEVER: while the solution structures between sornl2
# and sorh2 look slightly different, the solutions for
# sornl2 and sorh2u look absolutely identical.



Nx = 16
Ny = 16
cfl = 0.8
iter_max = 3000

time_compare(Nx, Ny, cfl, iter_max)
exit(0)

# method = "sornl2"
# omega = 1.60

# N = 16
# method = "sorh2"
# omega = 0.80

# N = 32 -> crashes at cfl ~ 1.323
# method = "sorh2"
# omega = 0.86

# N = 16
method = "sorh2_unsplit"
omega = 0.84

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

U, t, iter_count, res, err_conv = solve(method, Nx, Ny, cfl, iter_max, omega)

print("iterations:", iter_count)
print("final residual: ", res)
print("time t: ", t)

plot(U, Nx, Ny, err_conv)
