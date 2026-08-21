import boundary_conditions as bc
from methods.poisson.poisson_helpers import *
from methods.poisson.hu_methods import *


def _resolve_direction_vector(full_config):
    """
    Return bx, by based on config.
    """
    config = full_config.h_unsplit_config
    theta = config.theta
    bx = np.cos(theta)
    by = np.sin(theta)

    if getattr(config, "normalize_direction", True):
        target_norm = getattr(config, "direction_norm", np.sqrt(2.0))
        norm = np.sqrt(bx * bx + by * by)
        if norm == 0.0:
            raise ValueError("Direction vector (bx, by) must be nonzero.")
        bx *= target_norm / norm
        by *= target_norm / norm

    return bx, by


def _classify_quadrant(theta):
    theta_mod = theta % (2.0 * np.pi)

    if 0.0 <= theta_mod < 0.5 * np.pi:
        return "back_back"
    elif 0.5 * np.pi <= theta_mod < np.pi:
        return "forward_back"
    elif np.pi <= theta_mod < 1.5 * np.pi:
        return "forward_forward"
    else:
        return "back_forward"


def _directional_denominator(quadrant_mode, bxdy, bydx):
    if quadrant_mode == "back_back":
        return bxdy + bydx
    if quadrant_mode == "forward_back":
        return -bxdy + bydx
    if quadrant_mode == "forward_forward":
        return bxdy + bydx
    if quadrant_mode == "back_forward":
        return bxdy - bydx
    raise ValueError(f"Unknown directional quadrant_mode: {quadrant_mode}")


def get_params(config):
    """
    Compute derived parameters for the hyperbolic unsplit method.
    All mode/control choices must already be stored on config.
    """
    dx, dy, cfl = config.dx, config.dy, config.cfl

    dt_factor = getattr(config, "dt_factor", 1.0)
    dt = dt_factor * cfl * min(dx, dy)

    # default unsplit coefficients
    c1 = 1.0 / (dx + dy)
    c2 = dt * dx * dy

    bx, by = _resolve_direction_vector(config)
    bxdy = bx * dy
    bydx = by * dx

    hu_config = config.h_unsplit_config
    theta = hu_config.theta
    quadrant_mode = _classify_quadrant(theta)
    directional_den = None

    if hu_config.mode == "directional":
        if abs(bx) < 1e-14 or abs(by) < 1e-14:
            raise ValueError(
                "h_unsplit directional mode does not support axis-aligned "
                f"directions: bx={bx}, by={by}, theta={theta}"
            )

        directional_den = _directional_denominator(quadrant_mode, bxdy, bydx)
        if abs(directional_den) < 1e-14:
            raise ValueError(
                "h_unsplit directional denominator too small for "
                f"quadrant_mode='{quadrant_mode}': den={directional_den}"
            )

    if directional_den is None:
        c1r = 1.0 / (bxdy + bydx) if abs(bxdy + bydx) >= 1e-14 else 0.0
    else:
        c1r = 1.0 / directional_den
    update_coeffs = (bxdy, bydx, c1r, c2)

    params = {"dt": dt, "c1": c1, "c2": c2, "bx": bx, "by": by, "bxdy": bxdy, "bydx": bydx,
              "update_coeffs": update_coeffs, "theta": theta, "quadrant_mode": quadrant_mode}

    # validate weighted hybrid parameters if needed
    if hu_config.mode == "weighted_hybrid":
        alpha = hu_config.alpha
        beta = hu_config.beta
        eps_alpha = hu_config.eps_alpha

        if abs(alpha - 0.5) < eps_alpha or abs(beta - 0.5) < eps_alpha:
            raise ValueError(
                "For weighted_hybrid, alpha and beta must stay away from 0.5 "
                f"by at least eps_alpha={eps_alpha}."
            )

        den = (2.0 * alpha - 1.0) * bxdy + (2.0 * beta - 1.0) * bydx
        if abs(den) < 1e-14:
            raise ValueError(
                f"weighted_hybrid denominator too small: den = {den}"
            )

        params["alpha"] = alpha
        params["beta"] = beta
        params["eps_alpha"] = eps_alpha
        params["hybrid_den"] = den

    if hu_config.mode == "default2_blend":
        eta = hu_config.default2_blend_eta
        if not 0.0 <= eta <= 1.0:
            raise ValueError(
                "default2_blend_eta must satisfy 0 <= eta <= 1; "
                f"got {eta}."
            )
        params["default2_blend_eta"] = eta

    return params


def full_sweep(U, params, boundaries, config):
    """
    Execute one full sweep of h_unsplit according to config.h_unsplit_mode.
    """
    mode = config.h_unsplit_config.mode
    f_all = config.f_all
    update_coeffs = params["update_coeffs"]

    if mode == "default":
        U = unsplit_update(U, config, boundaries, (params["c1"], params["c2"]), f_all)

    elif mode == "red_black":
        U = red_black_rectangular(
            U,
            config,
            boundaries,
            (params["c1"], params["c2"]),
            f_all,
        )

    elif mode == "default2":
        U = back_x_back_y_2nd(U, config, boundaries, (params["c1"], params["c2"]), f_all)

    elif mode == "default2_blend":
        U = back_x_back_y_bdf_blend(
            U,
            config,
            boundaries,
            (params["c1"], params["c2"]),
            f_all,
            params["default2_blend_eta"],
        )

    elif mode == "weighted_hybrid":
        U = weighted_hybrid(
            U,
            config,
            boundaries,
            update_coeffs,
            f_all,
            params["alpha"],
            params["beta"],
        )

    elif mode == "directional":
        quadrant_mode = params["quadrant_mode"]

        if quadrant_mode == "back_back":
            U = back_x_back_y(U, config, boundaries, update_coeffs, f_all)
        elif quadrant_mode == "forward_back":
            U = forward_x_back_y(U, config, boundaries, update_coeffs, f_all)
        elif quadrant_mode == "forward_forward":
            U = forward_x_forward_y(U, config, boundaries, update_coeffs, f_all)
        elif quadrant_mode == "back_forward":
            U = back_x_forward_y(U, config, boundaries, update_coeffs, f_all)
        else:
            raise ValueError(f"Unknown directional quadrant_mode: {quadrant_mode}")

    else:
        raise ValueError(
            f"Unknown h_unsplit_mode='{mode}'. "
            "Expected one of: 'default', 'red_black', 'default2', "
            "'default2_blend', 'directional', 'weighted_hybrid'."
        )

    return U
