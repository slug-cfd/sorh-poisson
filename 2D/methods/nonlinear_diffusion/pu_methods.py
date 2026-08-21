"""Full-sweep kernels for nonlinear parabolic unsplit SOR."""

import boundary_conditions as bc
from methods.nonlinear_diffusion import nl_helpers as nlh


def _ordered_points(config):
    ib, ie, jb, je = config.ib, config.ie, config.jb, config.je
    ordering = config.p_unsplit_config.gs_ordering

    if ordering == "lexicographic":
        for i in range(ib, ie + 1):
            for j in range(jb, je + 1):
                yield i, j
        return
    if ordering == "red_black":
        for color in (0, 1):
            for i in range(ib, ie + 1):
                for j in range(jb, je + 1):
                    if (i + j) % 2 == color:
                        yield i, j
        return
    raise ValueError(f"Unknown p_unsplit gs_ordering='{ordering}'.")


def _set_timestep(params, config, vx=None, vy=None):
    if config.timestep_mode == "adaptive":
        if vx is None or vy is None:
            raise ValueError("Adaptive p_unsplit requires face coefficients.")
        dt = nlh.adaptive_parabolic_dt(vx, vy, config)
        rx = dt / config.dx**2
        ry = dt / config.dy**2
    else:
        dt = params.get("fixed_dt", params.get("dt", 0.0))
        rx = params.get("fixed_rx", params["rx"])
        ry = params.get("fixed_ry", params["ry"])
    params.update(dt=dt, rx=rx, ry=ry)
    return rx, ry


def _apply_candidate(U, U_old, i, j, rx, ry, omega, vxm, vxp, vym, vyp):
    x_rv = rx * (vxp + vxm)
    y_rv = ry * (vyp + vym)
    midpoint = 1.0 - x_rv - y_rv
    candidate = (
        midpoint * U[i, j]
        + rx * (vxm * U[i - 1, j] + vxp * U[i + 1, j])
        + ry * (vym * U[i, j - 1] + vyp * U[i, j + 1])
    )
    U[i, j] = U_old[i, j] + omega * (candidate - U_old[i, j])


def _pointwise_update(U, U_old, i, j, rx, ry, omega, config):
    vxm, vxp = nlh.get_vx_point(U, i, j, config)
    vym, vyp = nlh.get_vy_point(U, i, j, config)
    _apply_candidate(U, U_old, i, j, rx, ry, omega, vxm, vxp, vym, vyp)


def _frozen_update(U, U_old, vx, vy, i, j, rx, ry, omega, config):
    local_i = i - config.ib
    local_j = j - config.jb
    vxm = vx[local_i, local_j, 0]
    vxp = vx[local_i, local_j, 1]
    vym = vy[local_i, local_j, 0]
    vyp = vy[local_i, local_j, 1]
    _apply_candidate(U, U_old, i, j, rx, ry, omega, vxm, vxp, vym, vyp)


def pointwise_sweep(U, params, config):
    """Recompute nonlinear face coefficients from the evolving sweep state."""
    vx = vy = None
    if config.timestep_mode == "adaptive":
        vx = nlh.get_vx(U, config)
        vy = nlh.get_vy(U, config)
    rx, ry = _set_timestep(params, config, vx, vy)

    U_old = U.copy()
    for i, j in _ordered_points(config):
        _pointwise_update(U, U_old, i, j, rx, ry, config.omega, config)

    bc.update_BCs(U, config)
    return U


def frozen_coefficient_sweep(U, params, config):
    """Freeze every nonlinear face coefficient at the start of the sweep."""
    vx = nlh.get_vx(U, config)
    vy = nlh.get_vy(U, config)
    rx, ry = _set_timestep(params, config, vx, vy)

    U_old = U.copy()
    for i, j in _ordered_points(config):
        _frozen_update(
            U, U_old, vx, vy, i, j, rx, ry, config.omega, config
        )

    bc.update_BCs(U, config)
    return U
