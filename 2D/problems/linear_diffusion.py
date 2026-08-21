"""Solver dispatch for constant-coefficient linear diffusion."""

import numpy as np

from methods import linear_diffusion as linear_methods

def solve(params):
    match params.method:
        case "parabolic_jacobi":
            U, t, iter_count, res, err_conv = linear_methods.parabolic(params, iter_method="jacobi")
        case "parabolic_gs":
            U, t, iter_count, res, err_conv = linear_methods.parabolic(params, iter_method="gs")
        case "parabolic_sor":
            U, t, iter_count, res, err_conv = linear_methods.parabolic(params, iter_method="sor")
        case _:
            raise ValueError(f"Unknown linear diffusion method '{params.method}'.")

    history = err_conv[:iter_count]
    histories = {
        "update": history,
        "residual": history,
        "converged": res <= 1e-7,
        "stable": np.all(np.isfinite(history)),
        "reason": "tol_update" if res <= 1e-7 else "iter_max",
    }
    return U, t, iter_count, histories
