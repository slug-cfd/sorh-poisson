def get_params(config):
    v = config.linear_v
    dt = config.dt_factor * config.cfl * (config.dx**2) / (2.0 * v)
    r = v * dt / (config.dx**2)
    return {"v": v, "dt": dt, "r": r}


def full_sweep(U, params, config):
    r = params["r"]

    for i in range(config.ib, config.ie + 1):
        U[i] = r * U[i - 1] + (1.0 - 2.0 * r) * U[i] + r * U[i + 1]

    return U
