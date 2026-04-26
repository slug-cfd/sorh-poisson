from .helpers import nonlinear_face_values


def get_params(config):
    vmax = config.nonlinear_vmax
    dt = config.dt_factor * config.cfl * (config.dx**2) / (2.0 * vmax)
    r = dt / (config.dx**2)
    return {"vmax": vmax, "dt": dt, "r": r}


def full_sweep(U, params, config):
    r = params["r"]
    U_new = U.copy()
    dt_candidate = params["dt"]

    for i in range(config.ib, config.ie + 1):
        vm, vp = nonlinear_face_values(U, i)
        U_new[i] = r * vm * U[i - 1] + (1.0 - r * (vm + vp)) * U[i] + r * vp * U[i + 1]

        v_max = max(vm, vp)
        if abs(v_max) > 1e-2:
            dt_candidate = min(dt_candidate, config.dt_factor * config.cfl * (config.dx**2) / (2.0 * v_max))

    params["dt"] = dt_candidate
    return U_new
