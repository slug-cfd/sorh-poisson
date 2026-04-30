from methods.ToyProblem.nonlinear import nl_helpers as nlh
import BC_2D as bc


def get_params(config):
    cfl, dx, dy = config.cfl, config.dx, config.dy

    v = 0.5  # nonlinear -> maximum value for stability condition

    # dt = cfl * ((dx ** 2) / (2*v))  # 1D stability condition

    dt = cfl / (2 * v * (1 / dx ** 2 + 1 / dy ** 2))
    # equivalently: dt = cfl * (dx**2 * dy**2) / (2*v*(dx**2 + dy**2))

    # const. factors for method update rule
    rx = dt / (dx ** 2)
    ry = dt / (dy ** 2)

    params = {
        'v': v,
        'dt': dt,
        'rx': rx,
        'ry': ry,
    }

    return params


def full_sweep(U, params, config):
    # get parameters from config class and param dict
    omega, ib, ie, jb, je = config.omega, config.ib, config.ie, config.jb, config.je
    v, dt, rx, ry = params["v"], params["dt"], params["rx"], params["ry"]

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

    return U
