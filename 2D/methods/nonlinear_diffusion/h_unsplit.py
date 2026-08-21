from methods.nonlinear_diffusion import hu_methods


def get_params(config):
    v_max = 0.5
    dt = config.cfl * min(config.dx / v_max, config.dy / v_max)
    return {"v": v_max, "dt": dt, "fixed_dt": dt}


def full_sweep(U, params, config):
    return hu_methods.default_sweep(U, params, config)
