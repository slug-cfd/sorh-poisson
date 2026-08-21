from methods.nonlinear_diffusion import pu_methods


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
        'fixed_dt': dt,
        'fixed_rx': rx,
        'fixed_ry': ry,
    }

    return params


def full_sweep(U, params, config):
    coefficient_mode = config.p_unsplit_config.coefficient_mode
    sweep = {
        "pointwise": pu_methods.pointwise_sweep,
        "frozen_per_sweep": pu_methods.frozen_coefficient_sweep,
    }.get(coefficient_mode)
    if sweep is None:
        raise ValueError(
            f"Unknown p_unsplit coefficient_mode='{coefficient_mode}'."
        )
    return sweep(U, params, config)
