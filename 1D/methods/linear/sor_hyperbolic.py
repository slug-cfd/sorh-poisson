def get_params(config):
    v = config.hyperbolic_linear_v
    dt = config.dt_factor * config.cfl * (config.dx / v)
    r = v * dt / config.dx
    return {"v": v, "dt": dt, "r": r}


def full_sweep(U, params, config):
    r = params["r"]
    omega = config.omega
    U_old = U.copy()
    U_new = U.copy()

    for i in range(config.ib, config.ie + 1):
        proposed = (
            U_new[i - 1]
            + (-1.0 + r) * U[i - 1]
            + (1.0 - 2.0 * r) * U[i]
            + r * U[i + 1]
        )
        U_new[i] = U_old[i] + omega * (proposed - U_old[i])

    return U_new
