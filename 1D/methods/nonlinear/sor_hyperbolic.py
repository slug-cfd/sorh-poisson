from .helpers import nonlinear_face_values


def get_params(config):
    vmax = config.nonlinear_vmax
    dt = config.dt_factor * config.cfl * (config.dx / vmax)
    r = dt / config.dx
    return {"vmax": vmax, "dt": dt, "r": r}


def full_sweep(U, params, config):
    U_old = U.copy()
    U_new = U.copy()
    omega = config.omega
    dx = config.dx

    r = params["r"]
    vmax = params["vmax"]

    for i in range(config.ib, config.ie + 1):
        vm, vp = nonlinear_face_values(U, i)
        proposed = (
            U_new[i - 1]
            + (-1.0 + r * vm) * U[i - 1]
            + (1.0 - r * (vm + vp)) * U[i]
            + r * vp * U[i + 1]
        )
        U_new[i] = U_old[i] + omega * (proposed - U_old[i])

        vmax = max(vmax, abs(vm), abs(vp))

    dt = config.dt_factor * config.cfl * (dx / vmax)
    params["vmax"] = vmax
    params["dt"] = dt
    params["r"] = dt / dx

    return U_new
