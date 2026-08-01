from .helpers import nonlinear_face_values


def get_params(config):
    vmax = config.nonlinear_vmax
    dt = 0.5 * config.dt_factor * config.cfl * (config.dx**2) / vmax
    r = dt / (config.dx**2)
    return {"vmax": vmax, "dt": dt, "r": r}


def full_sweep(U, params, config):
    U_old = U.copy()
    omega = config.omega
    dx2 = config.dx**2

    r = params["r"]
    vmax = params["vmax"]

    for i in range(config.ib, config.ie + 1):
        vm, vp = nonlinear_face_values(U, i)

        proposed = (
            r * vm * U[i - 1]
            + (1.0 - r * (vm + vp)) * U[i]
            + r * vp * U[i + 1]
        )

        U[i] = U_old[i] + omega * (proposed - U_old[i])

        vmax = max(vmax, abs(vm), abs(vp))

    dt = 0.5 * config.dt_factor * config.cfl * dx2 / vmax
    params["vmax"] = vmax
    params["dt"] = dt
    params["r"] = dt / dx2

    return U
