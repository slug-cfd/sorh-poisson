"""Second-order dimensional splitting for nonlinear diffusion."""

import numpy as np
from methods.nonlinear_diffusion import ds_methods, nl_helpers as nlh
import boundary_conditions as bc


def get_params(config):
    # get parameters as needed
    Nx, Ny, dx, dy = config.Nx, config.Ny, config.dx, config.dy
    ib, ie, jb, je = config.ib, config.ie, config.jb, config.je
    cfl, iter_max = config.cfl, config.iter_max

    U = config.U0.copy()
    bc.update_BCs(U, config)

    ### Step 1: Initial 1/4-step ###
    vx = nlh.get_vx(U, config)

    dt = 0.5 * cfl * (dx / np.abs(vx).max())
    rx = dt / dx

    # initial x-sweep: U^n -> U^(n+1/4)
    U = ds_methods.x_sweep(U, vx, rx, config)
    # t += dt

    first_sweep = True
    last_sweep = False

    params = {
        "first_sweep": first_sweep,
        "last_sweep": last_sweep,
    }
    return params


def full_sweep(U, params, config):
    # get parameters as needed
    Nx, Ny, dx, dy = config.Nx, config.Ny, config.dx, config.dy
    cfl, iter_max = config.cfl, config.iter_max
    first_sweep, last_sweep = params["first_sweep"], params["last_sweep"]

    if first_sweep:
        ### Step 1: Initial 1/4-step ###
        vx = nlh.get_vx(U, config)

        dt = 0.5 * cfl * (dx / np.abs(vx).max())
        rx = dt / dx

        # initial x-sweep: U^n -> U^(n+1/4)
        U = ds_methods.x_sweep(U, vx, rx, config)

        params["first_sweep"] = False

    if last_sweep:
        ### Step 3: Final 3/4-step ###
        vx = nlh.get_vx(U, config)
        vy = nlh.get_vy(U, config)

        dt = cfl * np.min([dx / np.abs(vx).max(), dy / np.abs(vy).max()])

        ry = dt / dy
        rx = 0.5 * dt / dx  # half-step

        # final y-sweep: U^(n+1/4) -> U^(n+3/4)
        U = ds_methods.y_sweep(U, vy, ry, config)

        # final x-sweep: U^(n+3/4) -> U^(n+1)
        U = ds_methods.x_sweep(U, vx, rx, config)

        return U

    ### Step 2: WITHIN MAIN ITERATION LOOP/FULL SWEEP (full time steps) ###
    vx = nlh.get_vx(U, config)
    vy = nlh.get_vy(U, config)

    dt = cfl * np.min([dx / np.abs(vx).max(), dy / np.abs(vy).max()])
    ry = dt / dy
    rx = dt / dx

    # full y-sweep: U^(n+1/4) -> U^(n+3/4)
    U = ds_methods.y_sweep(U, vy, ry, config)

    # full x-sweep: U^(n+3/4) -> U^(n+5/4)
    U = ds_methods.x_sweep(U, vx, rx, config)

    return U
