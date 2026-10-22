"""Fourth-order linear SORh sweeps with selectable two-point causal LHS."""

from problem_helpers import fourth_order_laplacian_dimensionless


LHS_STENCILS = {
    "sor_linear_hyperbolic_fourth_comp": {
        "name": "P-comp",
        "c0": 15.0 / 14.0,
        "p1": 16.0 / 15.0,      # i-1
        "p2": -1.0 / 15.0,      # i-2
    },
    "sor_linear_hyperbolic_fourth_bdf2": {
        "name": "BDF2",
        "c0": 3.0 / 2.0,
        "p1": 4.0 / 3.0,        # i-1
        "p2": -1.0 / 3.0,       # i-2
    },
}


def get_params(config):
    v = config.hyperbolic_linear_v
    dt = config.dt_factor * config.cfl * (config.dx / v)
    r = v * dt / config.dx
    stencil = LHS_STENCILS[config.method]
    return {
        "v": v,
        "dt": dt,
        "r": r,
        "lhs_stencil": stencil["name"],
        "c0": stencil["c0"],
        "p1": stencil["p1"],
        "p2": stencil["p2"],
    }


def full_sweep(U, params, config):
    omega = config.omega
    residual_coefficient = params["r"] / params["c0"]
    p1 = params["p1"]
    p2 = params["p2"]

    U_old = U.copy()
    delta = U * 0.0

    for i in range(config.ib, config.ie + 1):
        laplacian_fourth = fourth_order_laplacian_dimensionless(U_old, i)
        delta[i] = omega * (
            p1 * delta[i - 1]
            + p2 * delta[i - 2]
            + residual_coefficient * laplacian_fourth
        )

    return U_old + delta
