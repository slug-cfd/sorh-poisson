import boundary_conditions as bc
import numpy as np

from methods.poisson.poisson_helpers import *


# default update
def unsplit_update(U, params, boundaries, update_coeffs, f_all):
    # U_old = U.copy()
    U_new = U.copy()

    Nx, Ny, dx, dy, omega = params.Nx, params.Ny, params.dx, params.dy, params.omega
    c1, c2 = update_coeffs
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je

    for i in range(ib, ie + 1):
        for j in range(jb, je + 1):
            U_new[i, j] = (U[i, j] + omega * c1 * ((dy * (U_new[i - 1, j] - U[i - 1, j]))
                                                   + (dx * (U_new[i, j - 1] - U[i, j - 1]))
                                                   + (c2 * (
                                laplacian_2D(U, i, j, params, params.laplacian_stencil) - f_all[i, j]))))

            # SOR update
            # U_new[i, j] = U_old[i, j] + (omega * (U_new[i, j] - U_old[i, j]))

    # update BCs
    bc.apply_bcs_poisson_2d(U_new, params, boundaries)

    return U_new


def red_black_rectangular(U, params, boundaries, update_coeffs, f_all):
    """Apply the coefficient-preserving rectangular-grid red-black SORh sweep.

    The Poisson residual is frozen at the beginning of the sweep.  The first
    color receives only the residual correction; the second color receives
    that same residual correction plus the four neighboring first-color
    corrections.  Keeping ``dx`` and ``dy`` separate preserves the directional
    weights in the original unsplit SORh discretization.  The supplied
    coefficients are ``c1 = 1 / (dx + dy)`` and
    ``c2 = pseudo_dt * dx * dy``, where
    ``pseudo_dt = dt_factor * CFL * min(dx, dy)``.

    This two-color form requires the bipartite centered five-point Laplacian.
    Wider or diagonal-coupling stencils need additional colors or a different
    block update and are therefore rejected here.
    """
    if params.laplacian_stencil != "centered":
        raise ValueError(
            "h_unsplit red_black mode requires "
            "laplacian_stencil='centered'."
        )

    U_old = U.copy()
    delta = np.zeros_like(U_old)

    c1, c2 = update_coeffs
    omega = params.omega
    dx, dy = params.dx, params.dy
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je

    residual_old = (
        get_all_laplace(U_old, params, params.laplacian_stencil) - f_all
    )

    residual_coeff = omega * c1 * c2
    x_propagation_coeff = omega * c1 * dy
    y_propagation_coeff = omega * c1 * dx

    # Color 0 has no same-sweep upstream correction.  Every neighbor of a
    # color-1 point belongs to color 0 for the centered five-point stencil.
    for i in range(ib, ie + 1):
        for j in range(jb, je + 1):
            if (i + j) % 2 == 0:
                delta[i, j] = residual_coeff * residual_old[i, j]

    for i in range(ib, ie + 1):
        for j in range(jb, je + 1):
            if (i + j) % 2 == 1:
                x_propagation = delta[i - 1, j] + delta[i + 1, j]
                y_propagation = delta[i, j - 1] + delta[i, j + 1]
                delta[i, j] = (
                    x_propagation_coeff * x_propagation
                    + y_propagation_coeff * y_propagation
                    + residual_coeff * residual_old[i, j]
                )

    U_new = U_old + delta
    bc.apply_bcs_poisson_2d(U_new, params, boundaries)

    return U_new


def forward_x_back_y(U, params, boundaries, update_coeffs, f_all):
    U_old = U.copy()
    U_new = U.copy()

    bxdy, bydx, c1, c2 = update_coeffs
    omega, ib, ie, jb, je = params.omega, params.ib, params.ie, params.jb, params.je
    den = -bxdy + bydx

    # forward in x, backward in y
    for i in range(ie, ib - 1, -1):
        for j in range(jb, je + 1):
            U_new[i, j] = U[i, j] + (
                    -bxdy * (U_new[i + 1, j] - U[i + 1, j])
                    + bydx * (U_new[i, j - 1] - U[i, j - 1])
                    + c2 * (laplacian_2D(U, i, j, params, params.laplacian_stencil) - f_all[i, j])
            ) / den

            # SOR update
            U_new[i, j] = U_old[i, j] + (omega * (U_new[i, j] - U_old[i, j]))

    # update BCs
    bc.apply_bcs_poisson_2d(U_new, params, boundaries)

    return U_new


def back_x_forward_y(U, params, boundaries, update_coeffs, f_all):
    U_old = U.copy()
    U_new = U.copy()

    bxdy, bydx, c1, c2 = update_coeffs
    omega, ib, ie, jb, je = params.omega, params.ib, params.ie, params.jb, params.je
    den = bxdy - bydx

    # backward in x, forward in y
    for i in range(ib, ie + 1):
        for j in range(je, jb - 1, -1):
            U_new[i, j] = U[i, j] + (
                    bxdy * (U_new[i - 1, j] - U[i - 1, j])
                    - bydx * (U_new[i, j + 1] - U[i, j + 1])
                    + c2 * (laplacian_2D(U, i, j, params, params.laplacian_stencil) - f_all[i, j])
            ) / den

            # SOR update
            U_new[i, j] = U_old[i, j] + (omega * (U_new[i, j] - U_old[i, j]))

    # update BCs
    bc.apply_bcs_poisson_2d(U_new, params, boundaries)

    return U_new


def forward_x_forward_y(U, params, boundaries, update_coeffs, f_all):
    U_old = U.copy()
    U_new = U.copy()

    bxdy, bydx, c1, c2 = update_coeffs
    omega, ib, ie, jb, je = params.omega, params.ib, params.ie, params.jb, params.je
    den = bxdy + bydx  # we will need to negate this for the update (both terms)

    # forward in both
    for i in range(ie, ib - 1, -1):
        for j in range(je, jb - 1, -1):
            U_new[i, j] = U[i, j] + (
                    bxdy * (U_new[i + 1, j] - U[i + 1, j])
                    + bydx * (U_new[i, j + 1] - U[i, j + 1])
                    - c2 * (laplacian_2D(U, i, j, params, params.laplacian_stencil) - f_all[i, j])
            ) / den

            # SOR update
            U_new[i, j] = U_old[i, j] + omega * (U_new[i, j] - U_old[i, j])

    # update BCs
    bc.apply_bcs_poisson_2d(U_new, params, boundaries)

    return U_new


def back_x_back_y(U, params, boundaries, update_coeffs, f_all):
    U_old = U.copy()
    U_new = U.copy()

    bxdy, bydx, c1, c2 = update_coeffs
    omega, ib, ie, jb, je = params.omega, params.ib, params.ie, params.jb, params.je

    # backward in both
    for i in range(ib, ie + 1):
        for j in range(jb, je + 1):
            U_new[i, j] = (U[i, j] + c1 * ((bxdy * (U_new[i - 1, j] - U[i - 1, j]))
                                           + (bydx * (U_new[i, j - 1] - U[i, j - 1]))
                                           + (c2 * (
                                laplacian_2D(U, i, j, params, params.laplacian_stencil) - f_all[i, j]))))

            # SOR update
            U_new[i, j] = U_old[i, j] + (omega * (U_new[i, j] - U_old[i, j]))

    # update BCs
    bc.apply_bcs_poisson_2d(U_new, params, boundaries)

    return U_new


def weighted_hybrid(U, params, boundaries, update_coeffs, f_all, alpha, beta):
    U_old = U.copy()
    U_new = U.copy()

    bxdy, bydx, _, c2 = update_coeffs
    omega, ib, ie, jb, je = params.omega, params.ib, params.ie, params.jb, params.je

    den = (2.0 * alpha - 1.0) * bxdy + (2.0 * beta - 1.0) * bydx

    # choose sweep directions from dominant side
    if alpha >= 0.5:
        i_range = range(ib, ie + 1)  # backward-dominant in x
    else:
        i_range = range(ie, ib - 1, -1)  # forward-dominant in x

    if beta >= 0.5:
        j_range = range(jb, je + 1)  # backward-dominant in y
    else:
        j_range = range(je, jb - 1, -1)  # forward-dominant in y

    for i in i_range:
        for j in j_range:
            # x-direction weighted increment:
            # use updated value only on the dominant side, lag the weaker side
            if alpha >= 0.5:
                delta_im1 = U_new[i - 1, j] - U[i - 1, j]
                x_inc = -alpha * delta_im1
            else:
                delta_ip1 = U_new[i + 1, j] - U[i + 1, j]
                x_inc = (1.0 - alpha) * delta_ip1

            # y-direction weighted increment
            if beta >= 0.5:
                delta_jm1 = U_new[i, j - 1] - U[i, j - 1]
                y_inc = -beta * delta_jm1
            else:
                delta_jp1 = U_new[i, j + 1] - U[i, j + 1]
                y_inc = (1.0 - beta) * delta_jp1

            rhs = c2 * (laplacian_2D(U, i, j, params, params.laplacian_stencil) - f_all[i, j])

            U_new[i, j] = U[i, j] + (rhs - bxdy * x_inc - bydx * y_inc) / den

            # SOR update
            U_new[i, j] = U_old[i, j] + omega * (U_new[i, j] - U_old[i, j])

    bc.apply_bcs_poisson_2d(U_new, params, boundaries)

    return U_new


def back_x_back_y_2nd(U, params, boundaries, update_coeffs, f_all):
    U_new = U.copy()

    c1, c2 = update_coeffs
    omega = params.omega
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
    dx, dy = params.dx, params.dy

    c1_2nd = c1 / 3.0
    stencil = params.laplacian_stencil

    # Startup row/column use BDF1 where the BDF2 upstream value is unavailable.
    if stencil == "centered":
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)

        for i in range(ib, ie + 1):
            x_has_bdf2 = i >= ib + 1
            for j in range(jb, je + 1):
                delta_im1 = U_new[i - 1, j] - U[i - 1, j]
                delta_jm1 = U_new[i, j - 1] - U[i, j - 1]
                y_has_bdf2 = j >= jb + 1

                u_ij = U[i, j]
                laplace = (
                        (U[i + 1, j] - 2.0 * u_ij + U[i - 1, j]) * inv_dx2
                        + (U[i, j + 1] - 2.0 * u_ij + U[i, j - 1]) * inv_dy2
                )
                rhs = c2 * (laplace - f_all[i, j])

                if x_has_bdf2 and y_has_bdf2:
                    delta_im2 = U_new[i - 2, j] - U[i - 2, j]
                    delta_jm2 = U_new[i, j - 2] - U[i, j - 2]
                    delta = c1_2nd * (
                            dy * (4.0 * delta_im1 - delta_im2)
                            + dx * (4.0 * delta_jm1 - delta_jm2)
                            + 2.0 * rhs
                    )
                elif x_has_bdf2:
                    delta_im2 = U_new[i - 2, j] - U[i - 2, j]
                    delta = (
                                    dy * (4.0 * delta_im1 - delta_im2)
                                    + 2.0 * dx * delta_jm1
                                    + 2.0 * rhs
                            ) / (3.0 * dy + 2.0 * dx)
                elif y_has_bdf2:
                    delta_jm2 = U_new[i, j - 2] - U[i, j - 2]
                    delta = (
                                    2.0 * dy * delta_im1
                                    + dx * (4.0 * delta_jm1 - delta_jm2)
                                    + 2.0 * rhs
                            ) / (2.0 * dy + 3.0 * dx)
                else:
                    delta = c1 * (
                            dy * delta_im1
                            + dx * delta_jm1
                            + rhs
                    )

                U_new[i, j] = u_ij + omega * delta
    else:
        for i in range(ib, ie + 1):
            x_has_bdf2 = i >= ib + 1
            for j in range(jb, je + 1):
                delta_im1 = U_new[i - 1, j] - U[i - 1, j]
                delta_jm1 = U_new[i, j - 1] - U[i, j - 1]
                y_has_bdf2 = j >= jb + 1
                rhs = c2 * (laplacian_2D(U, i, j, params, stencil) - f_all[i, j])

                if x_has_bdf2 and y_has_bdf2:
                    delta_im2 = U_new[i - 2, j] - U[i - 2, j]
                    delta_jm2 = U_new[i, j - 2] - U[i, j - 2]
                    delta = c1_2nd * (
                            dy * (4.0 * delta_im1 - delta_im2)
                            + dx * (4.0 * delta_jm1 - delta_jm2)
                            + 2.0 * rhs
                    )
                elif x_has_bdf2:
                    delta_im2 = U_new[i - 2, j] - U[i - 2, j]
                    delta = (
                                    dy * (4.0 * delta_im1 - delta_im2)
                                    + 2.0 * dx * delta_jm1
                                    + 2.0 * rhs
                            ) / (3.0 * dy + 2.0 * dx)
                elif y_has_bdf2:
                    delta_jm2 = U_new[i, j - 2] - U[i, j - 2]
                    delta = (
                                    2.0 * dy * delta_im1
                                    + dx * (4.0 * delta_jm1 - delta_jm2)
                                    + 2.0 * rhs
                            ) / (2.0 * dy + 3.0 * dx)
                else:
                    delta = c1 * (
                            dy * delta_im1
                            + dx * delta_jm1
                            + rhs
                    )

                U_new[i, j] = U[i, j] + omega * delta

    # update BCs
    bc.apply_bcs_poisson_2d(U_new, params, boundaries)

    return U_new


def back_x_back_y_bdf_blend(U, params, boundaries, update_coeffs, f_all, eta):
    U_new = U.copy()

    c1, c2 = update_coeffs
    omega = params.omega
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
    dx, dy = params.dx, params.dy

    w0 = 1.0 + 0.5 * eta
    w1 = 1.0 + eta
    w2 = 0.5 * eta
    inv_den_full = 1.0 / (w0 * (dx + dy))
    inv_den_x_blend = 1.0 / (w0 * dy + dx)
    inv_den_y_blend = 1.0 / (dy + w0 * dx)
    stencil = params.laplacian_stencil

    if stencil == "centered":
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)

        for i in range(ib, ie + 1):
            x_has_blend = i >= ib + 1
            for j in range(jb, je + 1):
                delta_im1 = U_new[i - 1, j] - U[i - 1, j]
                delta_jm1 = U_new[i, j - 1] - U[i, j - 1]
                y_has_blend = j >= jb + 1

                u_ij = U[i, j]
                laplace = (
                        (U[i + 1, j] - 2.0 * u_ij + U[i - 1, j]) * inv_dx2
                        + (U[i, j + 1] - 2.0 * u_ij + U[i, j - 1]) * inv_dy2
                )
                rhs = c2 * (laplace - f_all[i, j])

                if x_has_blend and y_has_blend:
                    delta_im2 = U_new[i - 2, j] - U[i - 2, j]
                    delta_jm2 = U_new[i, j - 2] - U[i, j - 2]
                    delta = inv_den_full * (
                            dy * (w1 * delta_im1 - w2 * delta_im2)
                            + dx * (w1 * delta_jm1 - w2 * delta_jm2)
                            + rhs
                    )
                elif x_has_blend:
                    delta_im2 = U_new[i - 2, j] - U[i - 2, j]
                    delta = inv_den_x_blend * (
                            dy * (w1 * delta_im1 - w2 * delta_im2)
                            + dx * delta_jm1
                            + rhs
                    )
                elif y_has_blend:
                    delta_jm2 = U_new[i, j - 2] - U[i, j - 2]
                    delta = inv_den_y_blend * (
                            dy * delta_im1
                            + dx * (w1 * delta_jm1 - w2 * delta_jm2)
                            + rhs
                    )
                else:
                    delta = c1 * (
                            dy * delta_im1
                            + dx * delta_jm1
                            + rhs
                    )

                U_new[i, j] = u_ij + omega * delta
    else:
        for i in range(ib, ie + 1):
            x_has_blend = i >= ib + 1
            for j in range(jb, je + 1):
                delta_im1 = U_new[i - 1, j] - U[i - 1, j]
                delta_jm1 = U_new[i, j - 1] - U[i, j - 1]
                y_has_blend = j >= jb + 1
                rhs = c2 * (laplacian_2D(U, i, j, params, stencil) - f_all[i, j])

                if x_has_blend and y_has_blend:
                    delta_im2 = U_new[i - 2, j] - U[i - 2, j]
                    delta_jm2 = U_new[i, j - 2] - U[i, j - 2]
                    delta = inv_den_full * (
                            dy * (w1 * delta_im1 - w2 * delta_im2)
                            + dx * (w1 * delta_jm1 - w2 * delta_jm2)
                            + rhs
                    )
                elif x_has_blend:
                    delta_im2 = U_new[i - 2, j] - U[i - 2, j]
                    delta = inv_den_x_blend * (
                            dy * (w1 * delta_im1 - w2 * delta_im2)
                            + dx * delta_jm1
                            + rhs
                    )
                elif y_has_blend:
                    delta_jm2 = U_new[i, j - 2] - U[i, j - 2]
                    delta = inv_den_y_blend * (
                            dy * delta_im1
                            + dx * (w1 * delta_jm1 - w2 * delta_jm2)
                            + rhs
                    )
                else:
                    delta = c1 * (
                            dy * delta_im1
                            + dx * delta_jm1
                            + rhs
                    )

                U_new[i, j] = U[i, j] + omega * delta

    bc.apply_bcs_poisson_2d(U_new, params, boundaries)

    return U_new
