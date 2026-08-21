"""Full-sweep kernels for nonlinear hyperbolic unsplit SOR."""

import boundary_conditions as bc
from methods.nonlinear_diffusion import nl_helpers as nlh


def default_sweep(U, params, config):
    """Apply the established lexicographic nonlinear SORh sweep."""
    omega, ib, ie, jb, je = (
        config.omega,
        config.ib,
        config.ie,
        config.jb,
        config.je,
    )
    dx, dy = config.dx, config.dy

    vx = nlh.get_vx(U, config)
    vy = nlh.get_vy(U, config)
    if config.timestep_mode == "adaptive":
        dt = nlh.adaptive_hyperbolic_dt(vx, vy, config)
    else:
        dt = params["fixed_dt"]
    params["dt"] = dt
    rx = dt * dy / dx
    ry = dt * dx / dy
    c = 1.0 / (dx + dy)

    U_new = U.copy()
    for i in range(ib, ie + 1):
        for j in range(jb, je + 1):
            local_i = i - ib
            local_j = j - jb
            vxm = vx[local_i, local_j, 0]
            vxp = vx[local_i, local_j, 1]
            vym = vy[local_i, local_j, 0]
            vyp = vy[local_i, local_j, 1]

            rxvp = rx * vxp
            rxvm = rx * vxm
            ryvp = ry * vyp
            ryvm = ry * vym
            midpoint = rxvp + rxvm + ryvp + ryvm
            correction = c * (
                dy * U_new[i - 1, j]
                + dx * U_new[i, j - 1]
                - midpoint * U[i, j]
                + (rxvm - dy) * U[i - 1, j]
                + (ryvm - dx) * U[i, j - 1]
                + rxvp * U[i + 1, j]
                + ryvp * U[i, j + 1]
            )
            U_new[i, j] = U[i, j] + omega * correction

    bc.update_BCs(U_new, config)
    return U_new
