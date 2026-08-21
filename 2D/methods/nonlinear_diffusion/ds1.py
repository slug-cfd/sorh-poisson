"""First-order dimensional splitting for nonlinear diffusion."""

import numpy as np
from methods.nonlinear_diffusion import ds_methods, nl_helpers as nlh


def get_params(config):
    params = {}
    return params


def full_sweep(U, params, config):
    # get parameters as needed
    Nx, Ny, dx, dy = config.Nx, config.Ny, config.dx, config.dy
    cfl, iter_max = config.cfl, config.iter_max

    vx = nlh.get_vx(U, config)
    vy = nlh.get_vy(U, config)

    # use a single dt per timestep
    dt = cfl * np.min([dx / np.abs(vx).max(), dy / np.abs(vy).max()])
    rx = dt / dx
    ry = dt / dy

    # 1. Solve u_tx = vu_xx: U^n -> U^(n+1/2)
    U = ds_methods.x_sweep(U, vx, rx, config)

    # 2. Solve u_ty = vu_yy: U^(n+1/2) -> U^(n+1)
    U = ds_methods.y_sweep(U, vy, ry, config)

    return U
