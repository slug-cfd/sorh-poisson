import numpy as np
import BC_2D as bc

def jacobi_sweep(U, params, coeffs):
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
    rx, ry, midpoint = coeffs

    for i in range(ib, ie + 1):
        for j in range(jb, je + 1):
            U[i, j] = midpoint * U[i, j] \
                      + rx * (U[i + 1, j] + U[i - 1, j]) \
                      + ry * (U[i, j + 1] + U[i, j - 1])

def sor_gs_sweep(U, U_old, params, coeffs, sor=True):
    ib, ie, jb, je, omega = params.ib, params.ie, params.jb, params.je, params.omega
    rx, ry, midpoint = coeffs

    # RED sweep
    for i in range(ib, ie + 1):
        for j in range(jb, je + 1):
            if (i + j) % 2 == 0:
                U[i, j] = midpoint * U[i, j] \
                          + rx * (U[i + 1, j] + U[i - 1, j]) \
                          + ry * (U[i, j + 1] + U[i, j - 1])

                # SOR update
                if sor:
                    U[i, j] = U_old[i, j] + (omega * (U[i, j] - U_old[i, j]))

    # BLACK sweep
    for i in range(ib, ie + 1):
        for j in range(jb, je + 1):
            if (i + j) % 2 == 1:
                U[i, j] = midpoint * U[i, j] \
                          + rx * (U[i + 1, j] + U[i - 1, j]) \
                          + ry * (U[i, j + 1] + U[i, j - 1])

                # SOR update
                if sor:
                    U[i, j] = U_old[i, j] + (omega * (U[i, j] - U_old[i, j]))

def parabolic(params, iter_method="sor"):
    # get parameters from config class
    cfl, dx, dy, Nx, Ny = params.cfl, params.dx, params.dy, params.Nx, params.Ny
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
    iter_max = params.iter_max

    v = 0.5     # linear -> const. v

    # dt = cfl * ((dx ** 2) / (2*v))  # 1D stability condition

    dt = cfl / (2 * v * (1 / dx ** 2 + 1 / dy ** 2))
    # equivalently: dt = cfl * (dx**2 * dy**2) / (2*v*(dx**2 + dy**2))

    # const. factors for method update rule
    rx = v * dt / (dx ** 2)
    ry = v * dt / (dy ** 2)
    midpoint = 1.0 - (2.0 * rx) - (2.0 * ry)
    coeffs = (rx, ry, midpoint)

    U = params.U0.copy()
    bc.update_BCs(U, params)

    #  error array (one value per iteration)
    err_conv = np.zeros(iter_max)

    t = 0.0
    iter_count = 0

    threshold = 1e-7
    res = 1e2

    while res > threshold and iter_count < iter_max:
        #  store the old solution to measure the steadiness
        U_old = U.copy()

        if iter_method == "sor":
            sor_gs_sweep(U, U_old, params, coeffs, sor=True)
        elif iter_method == "gs":
            sor_gs_sweep(U, U_old, params, coeffs, sor=False)
        elif iter_method == "jacobi":
            jacobi_sweep(U, params, coeffs)
        else:
            print("Invalid iteration method")
            return 0

        # update BCs
        bc.update_BCs(U, params)

        # calculate error
        err = U - U_old

        res = np.linalg.norm(err[ib:ie+1, jb:je+1], ord="fro") / np.sqrt(Nx * Ny)
        err_conv[iter_count] = res

        #  update clock and iteration count
        t = t + dt
        iter_count += 1

    # pointwise error (linear -> exact solution = x)
    # err_exact = np.abs(x - U)

    # return U, t, iter_count, res, err_conv, err_exact
    return U, t, iter_count, res, err_conv

# def x_sweep(U, vx, rx, params):
#     Nx, Ny, omega = params.Nx, params.Ny, params.omega
#     ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
#
#     U_old = U.copy()
#     U_new = U.copy()
#
#     for j in range(jb, je+1):  # y-loop: no y-dependence for x-sweep
#         for i in range(ib, ie+1):  # x-loop: 1D hyperbolic update rule
#             # get values for v
#             vm = vx[i-1, j-1, 0]
#             vp = vx[i-1, j-1, 1]
#
#             # 1D hyperbolic discretization
#             U_new[i, j] = (U_new[i - 1, j] + ((-1 + rx * vm) * U[i - 1, j])
#                            + ((1 - rx * (vm + vp)) * U[i, j] + (rx * vp) * U[i + 1, j]))
#
#             # SOR update
#             U_new[i, j] = U_old[i, j] + (omega * (U_new[i, j] - U_old[i, j]))
#
#     bc.update_BCs(U_new, params)
#     return U_new
#
# def y_sweep(U, vy, ry, params):
#     Nx, Ny, omega = params.Nx, params.Ny, params.omega
#     ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
#
#     U_old = U.copy()
#     U_new = U.copy()
#
#     for i in range(ib, ie+1):  # x-loop: no x-dependence for y-sweep
#         for j in range(jb, je+1):  # y-loop: 1D hyperbolic update rule
#             # get values for v
#             vm = vy[i-1, j-1, 0]
#             vp = vy[i-1, j-1, 1]
#
#             # 1D hyperbolic discretization
#             U_new[i, j] = (U_new[i, j - 1] + ((-1 + ry * vm) * U[i, j - 1])
#                            + ((1 - ry * (vm + vp)) * U[i, j] + (ry * vp) * U[i, j + 1]))
#
#             # SOR update
#             U_new[i, j] = U_old[i, j] + (omega * (U_new[i, j] - U_old[i, j]))
#
#     bc.update_BCs(U_new, params)
#     return U_new
#
# def DS1(params):
#     # get parameters as needed
#     Nx, Ny, dx, dy = params.Nx, params.Ny, params.dx, params.dy
#     ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
#     cfl, iter_max = params.cfl, params.iter_max
#
#     U = params.U0.copy()
#     bc.update_BCs(U, params)
#
#     #  error array (one value per iteration)
#     err_conv = np.zeros(iter_max)
#
#     t = 0.0
#     iter_count = 0
#
#     threshold = 1e-7
#     res = 1e2
#
#     while res > threshold and iter_count < iter_max:
#         U_init = U.copy()
#
#         vx = nlh.get_vx(U, params)
#         vy = nlh.get_vy(U, params)
#
#         # use a single dt per timestep
#         dt = cfl * np.min([dx / np.abs(vx).max(), dy / np.abs(vy).max()])
#         rx = dt / dx
#         ry = dt / dy
#
#         # 1. Solve u_tx = vu_xx: U^n -> U^(n+1/2)
#         U = x_sweep(U, vx, rx, params)
#
#         # 2. Solve u_ty = vu_yy: U^(n+1/2) -> U^(n+1)
#         U = y_sweep(U, vy, ry, params)
#
#         # calculate error (one full cycle)
#         err = U - U_init
#
#         res = np.linalg.norm(err[ib:ie+1, jb:je+1], ord="fro") / np.sqrt(Nx * Ny)
#         err_conv[iter_count] = res
#
#         #  update clock and iteration count
#         t = t + dt
#         iter_count += 1
#
#     return U, t, iter_count, res, err_conv
#
# def DS2(params):
#     # get parameters as needed
#     Nx, Ny, dx, dy = params.Nx, params.Ny, params.dx, params.dy
#     ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
#     cfl, iter_max = params.cfl, params.iter_max
#
#     U = params.U0.copy()
#     bc.update_BCs(U, params)
#
#     # error array (one value per iteration)
#     err_conv = np.zeros(iter_max)
#
#     t = 0.0
#     iter_count = 0
#
#     threshold = 1e-7
#     res = 1e2
#
#     ### Step 1: Initial 1/4-step ###
#     vx = nlh.get_vx(U, params)
#
#     dt = 0.5 * cfl * (dx / np.abs(vx).max())
#     rx = dt / dx
#
#     # initial x-sweep: U^n -> U^(n+1/4)
#     U = x_sweep(U, vx, rx, params)
#     t += dt
#
#
#     ### Step 2: main iteration loop (full time steps) ###
#     while res > threshold and iter_count < iter_max:
#         U_init = U.copy()
#
#         vx = nlh.get_vx(U, params)
#         vy = nlh.get_vy(U, params)
#
#         dt = cfl * np.min([dx / np.abs(vx).max(), dy / np.abs(vy).max()])
#         ry = dt / dy
#         rx = dt / dx
#
#         # full y-sweep: U^(n+1/4) -> U^(n+3/4)
#         U = y_sweep(U, vy, ry, params)
#
#         # full x-sweep: U^(n+3/4) -> U^(n+5/4)
#         U = x_sweep(U, vx, rx, params)
#
#         # calculate error (one full cycle)
#         err = U - U_init
#
#         res = np.linalg.norm(err[ib:ie+1, jb:je+1], ord="fro") / np.sqrt(Nx * Ny)
#         err_conv[iter_count] = res
#
#         #  update clock and iteration count
#         t = t + dt
#         iter_count += 1
#
#     ### Step 3: Final 3/4-step ###
#     vx = nlh.get_vx(U, params)
#     vy = nlh.get_vy(U, params)
#
#     dt = cfl * np.min([dx / np.abs(vx).max(), dy / np.abs(vy).max()])
#
#     ry = dt / dy
#     rx = 0.5 * dt / dx  # half-step
#
#     # final y-sweep: U^(n+1/4) -> U^(n+3/4)
#     U = y_sweep(U, vy, ry, params)
#
#     # final x-sweep: U^(n+3/4) -> U^(n+1)
#     U = x_sweep(U, vx, rx, params)
#
#     # since dt is different between step 1 and step 3, we
#     # estimate the total dt value between both iterations as
#     # 0.5 * (dt_init + dt_final)
#     t += 0.5 * dt
#
#     # extra iteration to account for step 1 and step 3
#     iter_count += 1
#
#     return U, t, iter_count, res, err_conv
#
# def unsplit_update(U, params, update_coeffs):
#     U_old = U.copy()
#     U_new = U.copy()
#
#     Nx, Ny, dx, dy, omega = params.Nx, params.Ny, params.dx, params.dy, params.omega
#     ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
#
#     vx, vy, rx, ry = update_coeffs
#
#     # only update interior cells: [ib, ie]
#     for i in range(ib, ie + 1):
#         for j in range(jb, je + 1):
#             # get values for v
#             vxm = vx[i - 1, j - 1, 0]
#             vxp = vx[i - 1, j - 1, 1]
#             vym = vy[i - 1, j - 1, 0]
#             vyp = vy[i - 1, j - 1, 1]
#
#             rxvp = rx * vxp
#             rxvm = rx * vxm
#             ryvp = ry * vyp
#             ryvm = ry * vym
#
#             midpoint = rxvp + rxvm + ryvp + ryvm
#             c = 1.0 / (dx + dy)
#
#             U_new[i, j] = U[i, j] + c * (dy * U_new[i - 1, j] + dx * U_new[i, j - 1]
#                                          + (-midpoint * U[i, j]) + ((rxvm - dy) * U[i - 1, j])
#                                          + ((ryvm - dx) * U[i, j - 1]) + (rxvp * U[i + 1, j])
#                                          + (ryvp * U[i, j + 1]))
#
#             # SOR update
#             U_new[i, j] = U_old[i, j] + (omega * (U_new[i, j] - U_old[i, j]))
#
#     bc.update_BCs(U_new, params)
#     return U_new
#
# def unsplit(params):
#     # get parameters from config class
#     cfl, dx, dy, Nx, Ny = params.cfl, params.dx, params.dy, params.Nx, params.Ny
#     ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
#
#     U = params.U0.copy()
#     bc.update_BCs(U, params)
#
#     # error array (one value per iteration)
#     err_conv = np.zeros(params.iter_max)
#
#     t = 0.0
#     iter_count = 0
#
#     threshold = 1e-7
#     res = 1e2
#
#     while res > threshold and iter_count < params.iter_max:
#         U_init = U.copy()
#
#         vx = nlh.get_vx(U, params)
#         vy = nlh.get_vy(U, params)
#
#         # use a single dt per timestep
#         dt = cfl * np.min([dx / np.abs(vx).max(), dy / np.abs(vy).max()])
#         rx = dt*dy / dx
#         ry = dt*dx / dy
#
#         update_coeffs = (vx, vy, rx, ry)
#
#         # update U^n -> U^n+1
#         U = unsplit_update(U, params, update_coeffs)
#
#         # calculate error (one full cycle)
#         err = U - U_init
#
#         res = np.linalg.norm(err[ib:ie+1, jb:je+1], ord="fro") / np.sqrt(Nx * Ny)
#         err_conv[iter_count] = res
#
#         #  update clock and iteration count
#         t = t + dt
#         iter_count += 1
#
#     return U, t, iter_count, res, err_conv

