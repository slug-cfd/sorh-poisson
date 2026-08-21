"""Boundary-condition helpers shared by the 2D solvers."""

import numpy as np
from solver_utils import (
    apply_dirichlet_boundaries,
    get_dirichlet_boundaries,
)

def update_BCs(U, config, phi_exact=None):
    a, b, c, d = 1.0, 2.0, 1.0, 2.0

    # update BCs
    match config.bc_type:
        case "poisson":
            boundaries = phi_exact if phi_exact is not None else get_bcs_poisson_2d(config, config.phi_exact)
            apply_bcs_poisson_2d(U, config, boundaries)
        case "linear_in_x":
            if config.problem == "nonlinear_diffusion":
                boundaries = getattr(config, "_nonlinear_boundaries", None)
                if boundaries is None:
                    boundaries = get_dirichlet_boundaries(
                        config,
                        config.problem_instance.boundary,
                    )
                    config._nonlinear_boundaries = boundaries
                apply_dirichlet_boundaries(U, config, boundaries)
            else:
                linear_in_x(U, config.x, config.ngc)
        case "other_1":
            # BC: spatial oscillations
            coeffs = [a, b, c, d]
            sinusoidal(U, config.x, coeffs)
        case "other_2":
            # BC: combination (mixed nonlinear functions)
            coeffs = [a, b]
            combination(U, config.x, coeffs)
        case _:
            print("Unrecognized bc_type: ", config.bc_type)


def get_bcs_poisson_2d(config, phi_exact):
    """
    Return fixed Dirichlet boundary/ghost layers for the stored grid.

    Nx and Ny count interior unknowns.  The stored grid has shape
    (Nx + 2*ngc, Ny + 2*ngc), and solver updates are restricted to
    config.ib:config.ie and config.jb:config.je.
    """
    return get_dirichlet_boundaries(config, config._eval_exact)

def apply_bcs_poisson_2d(U, config, boundaries):
    return apply_dirichlet_boundaries(U, config, boundaries)


# Dirichlet boundary profile: U(x, y) = x
def linear_in_x(U, x, ngc=1):
    # Apply the exact profile u(x, y) = x on every fixed boundary/ghost layer.
    U[:ngc, :] = x[:ngc, None]
    U[-ngc:, :] = x[-ngc:, None]
    U[:, :ngc] = x[:, None]
    U[:, -ngc:] = x[:, None]

    return U

def sinusoidal(U, x, config):
    a, b, c, d = config

    U[:, 0] = a * np.sin(b * np.pi * x)
    U[:, -1] = 0.0
    U[0, :] = c * np.sin(d * np.pi * x)
    U[-1, :] = 0.0

    return U


def combination(U, x, config):
    a, b = config

    U[:, 0] = a * np.sin(b * np.pi * x)
    U[:, -1] = 0.0
    U[0, :] = -np.sqrt(x)
    U[-1, :] = np.sqrt(x)

    return U
