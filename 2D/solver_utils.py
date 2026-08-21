import numpy as np


def normalized_grid_norm(values, norm):
    """Return a vector norm over all supplied grid values."""
    flat = np.asarray(values).ravel()
    if flat.size == 0:
        return 0.0

    ord_value = 2 if norm == "fro" else norm
    result = float(np.linalg.norm(flat, ord=ord_value))
    if norm in (1, 2, "fro"):
        result /= np.sqrt(flat.size)
    return result


def stopping_reason(update_residual, equation_residual, stopping_config):
    """Return a terminal reason, or None when iteration should continue."""
    values = np.array([update_residual, equation_residual], dtype=float)
    if not np.all(np.isfinite(values)):
        return "nonfinite"
    if np.max(np.abs(values)) > stopping_config.res_max:
        return "res_max"
    if equation_residual < stopping_config.tol_residual:
        return "tol_residual"
    if update_residual < stopping_config.tol_update:
        return "stagnated_update"
    return None


def get_dirichlet_boundaries(config, boundary_function):
    """Evaluate fixed Dirichlet data on every stored boundary/ghost layer."""
    x, y, ngc = config.x, config.y, config.ngc

    def evaluate(x_values, y_values):
        try:
            return np.asarray(boundary_function(x_values, y_values, config))
        except TypeError:
            return np.asarray(boundary_function(x_values, y_values))

    x_left = np.array([evaluate(x[g], y) for g in range(ngc)])
    x_right = np.array([evaluate(x[-ngc + g], y) for g in range(ngc)])
    y_bottom = np.array([evaluate(x, y[g]) for g in range(ngc)])
    y_top = np.array([evaluate(x, y[-ngc + g]) for g in range(ngc)])
    return x_left, x_right, y_bottom, y_top


def apply_dirichlet_boundaries(U, config, boundaries):
    """Apply precomputed Dirichlet data to every fixed outer layer."""
    ngc = config.ngc
    x_left, x_right, y_bottom, y_top = boundaries

    for g in range(ngc):
        U[g, :] = x_left[g]
        U[-ngc + g, :] = x_right[g]
        U[:, g] = y_bottom[g]
        U[:, -ngc + g] = y_top[g]

    return U
