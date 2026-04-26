from .helpers import nonlinear_face_values


def get_params(config):
    vmax = config.nonlinear_vmax
    dt = config.dt_factor * config.cfl * (config.dx / vmax)
    r = dt / config.dx
    return {"vmax": vmax, "dt": dt, "r": r}


def full_sweep(U, params, config):
    r = params["r"]
    omega = config.omega
    U_old = U.copy()
    U_new = U.copy()
    dt_candidate = params["dt"]

    for i in range(config.ib, config.ie + 1):
        vm, vp = nonlinear_face_values(U, i)
        proposed = (
            U_new[i - 1]
            + (-1.0 + r * vm) * U[i - 1]
            + (1.0 - r * (vm + vp)) * U[i]
            + r * vp * U[i + 1]
        )
        U_new[i] = U_old[i] + omega * (proposed - U_old[i])

        v_max = max(vm, vp)
        if abs(v_max) > 1e-2:
            dt_candidate = min(dt_candidate, config.dt_factor * config.cfl * (config.dx / v_max))

    params["dt"] = dt_candidate
    return U_new
