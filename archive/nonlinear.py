import numpy as np
import methods.ToyProblem.nonlinear.nl_helpers as nlh
import BC_2D as bc

def parabolic(config):
    # get parameters from config class
    cfl, dx, dy, Nx, Ny = config.cfl, config.dx, config.dy, config.Nx, config.Ny
    ib, ie, jb, je = config.ib, config.ie, config.jb, config.je
    omega, iter_max = config.omega, config.iter_max

    v = 0.5     # nonlinear -> maximum value for stability condition

    # dt = cfl * ((dx ** 2) / (2*v))  # 1D stability condition

    dt = cfl / (2 * v * (1 / dx ** 2 + 1 / dy ** 2))
    # equivalently: dt = cfl * (dx**2 * dy**2) / (2*v*(dx**2 + dy**2))

    # const. factors for method update rule
    rx = dt / (dx ** 2)
    ry = dt / (dy ** 2)

    U = config.U0.copy()
    bc.update_BCs(U, config)

    #  error array (one value per iteration)
    err_conv = np.zeros(iter_max)

    t = 0.0
    iter_count = 0

    threshold = 1e-7
    res = 1e2

    while res > threshold and iter_count < iter_max:
        #  store the old solution to measure the steadiness
        U_old = U.copy()

        vx = nlh.get_vx(U, config)
        vy = nlh.get_vy(U, config)

        # RED sweep
        for i in range(ib, ie + 1):
            for j in range(jb, je + 1):
                if (i + j) % 2 == 0:
                    # get values for v
                    vxm = vx[i - 1, j - 1, 0]
                    vxp = vx[i - 1, j - 1, 1]
                    vym = vy[i - 1, j - 1, 0]
                    vyp = vy[i - 1, j - 1, 1]

                    x_rv = rx * (vxp + vxm)
                    y_rv = ry * (vyp + vym)
                    midpoint = 1.0 - x_rv - y_rv

                    U[i, j] = midpoint * U[i, j] \
                                  + rx * (vxm * U[i - 1, j] + vxp * U[i + 1, j]) \
                                  + ry * (vym * U[i, j - 1] + vyp * U[i, j + 1])

                    # SOR update
                    U[i, j] = U_old[i, j] + (omega * (U[i, j] - U_old[i, j]))

        # BLACK sweep
        for i in range(ib, ie + 1):
            for j in range(jb, je + 1):
                if (i + j) % 2 == 1:
                    # get values for v
                    vxm = vx[i - 1, j - 1, 0]
                    vxp = vx[i - 1, j - 1, 1]
                    vym = vy[i - 1, j - 1, 0]
                    vyp = vy[i - 1, j - 1, 1]

                    x_rv = rx * (vxp + vxm)
                    y_rv = ry * (vyp + vym)
                    midpoint = 1.0 - x_rv - y_rv

                    U[i, j] = midpoint * U[i, j] \
                                  + rx * (vxm * U[i - 1, j] + vxp * U[i + 1, j]) \
                                  + ry * (vym * U[i, j - 1] + vyp * U[i, j + 1])

                    # SOR update
                    U[i, j] = U_old[i, j] + (omega * (U[i, j] - U_old[i, j]))

        # update BCs
        bc.update_BCs(U, config)

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

def x_sweep(U, vx, rx, config):
    Nx, Ny, omega = config.Nx, config.Ny, config.omega
    ib, ie, jb, je = config.ib, config.ie, config.jb, config.je

    U_old = U.copy()
    U_new = U.copy()

    for j in range(jb, je+1):  # y-loop: no y-dependence for x-sweep
        for i in range(ib, ie+1):  # x-loop: 1D hyperbolic update rule
            # get values for v
            vm = vx[i-1, j-1, 0]
            vp = vx[i-1, j-1, 1]

            # 1D hyperbolic discretization
            U_new[i, j] = (U_new[i - 1, j] + ((-1 + rx * vm) * U[i - 1, j])
                           + ((1 - rx * (vm + vp)) * U[i, j] + (rx * vp) * U[i + 1, j]))

            # SOR update
            U_new[i, j] = U_old[i, j] + (omega * (U_new[i, j] - U_old[i, j]))

    bc.update_BCs(U_new, config)
    return U_new

def y_sweep(U, vy, ry, config):
    Nx, Ny, omega = config.Nx, config.Ny, config.omega
    ib, ie, jb, je = config.ib, config.ie, config.jb, config.je

    U_old = U.copy()
    U_new = U.copy()

    for i in range(ib, ie+1):  # x-loop: no x-dependence for y-sweep
        for j in range(jb, je+1):  # y-loop: 1D hyperbolic update rule
            # get values for v
            vm = vy[i-1, j-1, 0]
            vp = vy[i-1, j-1, 1]

            # 1D hyperbolic discretization
            U_new[i, j] = (U_new[i, j - 1] + ((-1 + ry * vm) * U[i, j - 1])
                           + ((1 - ry * (vm + vp)) * U[i, j] + (ry * vp) * U[i, j + 1]))

            # SOR update
            U_new[i, j] = U_old[i, j] + (omega * (U_new[i, j] - U_old[i, j]))

    bc.update_BCs(U_new, config)
    return U_new

def DS1(config):
    # get parameters as needed
    Nx, Ny, dx, dy = config.Nx, config.Ny, config.dx, config.dy
    ib, ie, jb, je = config.ib, config.ie, config.jb, config.je
    cfl, iter_max = config.cfl, config.iter_max

    U = config.U0.copy()
    bc.update_BCs(U, config)

    #  error array (one value per iteration)
    err_conv = np.zeros(iter_max)

    t = 0.0
    iter_count = 0

    threshold = 1e-7
    res = 1e2

    while res > threshold and iter_count < iter_max:
        U_init = U.copy()

        vx = nlh.get_vx(U, config)
        vy = nlh.get_vy(U, config)

        # use a single dt per timestep
        dt = cfl * np.min([dx / np.abs(vx).max(), dy / np.abs(vy).max()])
        rx = dt / dx
        ry = dt / dy

        # 1. Solve u_tx = vu_xx: U^n -> U^(n+1/2)
        U = x_sweep(U, vx, rx, config)

        # 2. Solve u_ty = vu_yy: U^(n+1/2) -> U^(n+1)
        U = y_sweep(U, vy, ry, config)

        # calculate error (one full cycle)
        err = U - U_init

        res = np.linalg.norm(err[ib:ie+1, jb:je+1], ord="fro") / np.sqrt(Nx * Ny)
        err_conv[iter_count] = res

        #  update clock and iteration count
        t = t + dt
        iter_count += 1

    return U, t, iter_count, res, err_conv

def DS2(config):
    # get parameters as needed
    Nx, Ny, dx, dy = config.Nx, config.Ny, config.dx, config.dy
    ib, ie, jb, je = config.ib, config.ie, config.jb, config.je
    cfl, iter_max = config.cfl, config.iter_max

    U = config.U0.copy()
    bc.update_BCs(U, config)

    # error array (one value per iteration)
    err_conv = np.zeros(iter_max)

    t = 0.0
    iter_count = 0

    threshold = 1e-7
    res = 1e2

    ### Step 1: Initial 1/4-step ###
    vx = nlh.get_vx(U, config)

    dt = 0.5 * cfl * (dx / np.abs(vx).max())
    rx = dt / dx

    # initial x-sweep: U^n -> U^(n+1/4)
    U = x_sweep(U, vx, rx, config)
    t += dt


    ### Step 2: main iteration loop (full time steps) ###
    while res > threshold and iter_count < iter_max:
        U_init = U.copy()

        vx = nlh.get_vx(U, config)
        vy = nlh.get_vy(U, config)

        dt = cfl * np.min([dx / np.abs(vx).max(), dy / np.abs(vy).max()])
        ry = dt / dy
        rx = dt / dx

        # full y-sweep: U^(n+1/4) -> U^(n+3/4)
        U = y_sweep(U, vy, ry, config)

        # full x-sweep: U^(n+3/4) -> U^(n+5/4)
        U = x_sweep(U, vx, rx, config)

        # calculate error (one full cycle)
        err = U - U_init

        res = np.linalg.norm(err[ib:ie+1, jb:je+1], ord="fro") / np.sqrt(Nx * Ny)
        err_conv[iter_count] = res

        #  update clock and iteration count
        t = t + dt
        iter_count += 1

    ### Step 3: Final 3/4-step ###
    vx = nlh.get_vx(U, config)
    vy = nlh.get_vy(U, config)

    dt = cfl * np.min([dx / np.abs(vx).max(), dy / np.abs(vy).max()])

    ry = dt / dy
    rx = 0.5 * dt / dx  # half-step

    # final y-sweep: U^(n+1/4) -> U^(n+3/4)
    U = y_sweep(U, vy, ry, config)

    # final x-sweep: U^(n+3/4) -> U^(n+1)
    U = x_sweep(U, vx, rx, config)

    # since dt is different between step 1 and step 3, we
    # estimate the total dt value between both iterations as
    # 0.5 * (dt_init + dt_final)
    t += 0.5 * dt

    # extra iteration to account for step 1 and step 3
    iter_count += 1

    return U, t, iter_count, res, err_conv

def unsplit_update(U, config, update_coeffs):
    U_old = U.copy()
    U_new = U.copy()

    Nx, Ny, dx, dy, omega = config.Nx, config.Ny, config.dx, config.dy, config.omega
    ib, ie, jb, je = config.ib, config.ie, config.jb, config.je

    vx, vy, rx, ry = update_coeffs

    # only update interior cells: [ib, ie]
    for i in range(ib, ie + 1):
        for j in range(jb, je + 1):
            # get values for v
            vxm = vx[i - 1, j - 1, 0]
            vxp = vx[i - 1, j - 1, 1]
            vym = vy[i - 1, j - 1, 0]
            vyp = vy[i - 1, j - 1, 1]

            rxvp = rx * vxp
            rxvm = rx * vxm
            ryvp = ry * vyp
            ryvm = ry * vym

            midpoint = rxvp + rxvm + ryvp + ryvm
            c = 1.0 / (dx + dy)

            U_new[i, j] = U[i, j] + c * (dy * U_new[i - 1, j] + dx * U_new[i, j - 1]
                                         + (-midpoint * U[i, j]) + ((rxvm - dy) * U[i - 1, j])
                                         + ((ryvm - dx) * U[i, j - 1]) + (rxvp * U[i + 1, j])
                                         + (ryvp * U[i, j + 1]))

            # SOR update
            U_new[i, j] = U_old[i, j] + (omega * (U_new[i, j] - U_old[i, j]))

    bc.update_BCs(U_new, config)
    return U_new

def unsplit(config):
    # get parameters from config class
    cfl, dx, dy, Nx, Ny = config.cfl, config.dx, config.dy, config.Nx, config.Ny
    ib, ie, jb, je = config.ib, config.ie, config.jb, config.je

    U = config.U0.copy()
    bc.update_BCs(U, config)

    # error array (one value per iteration)
    err_conv = np.zeros(config.iter_max)

    t = 0.0
    iter_count = 0

    threshold = 1e-7
    res = 1e2

    while res > threshold and iter_count < config.iter_max:
        U_init = U.copy()

        vx = nlh.get_vx(U, config)
        vy = nlh.get_vy(U, config)

        # use a single dt per timestep
        dt = cfl * np.min([dx / np.abs(vx).max(), dy / np.abs(vy).max()])
        rx = dt*dy / dx
        ry = dt*dx / dy

        update_coeffs = (vx, vy, rx, ry)

        # update U^n -> U^n+1
        U = unsplit_update(U, config, update_coeffs)

        # calculate error (one full cycle)
        err = U - U_init

        res = np.linalg.norm(err[ib:ie+1, jb:je+1], ord="fro") / np.sqrt(Nx * Ny)
        err_conv[iter_count] = res

        #  update clock and iteration count
        t = t + dt
        iter_count += 1

    return U, t, iter_count, res, err_conv

