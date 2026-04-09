import BC_2D as bc
from methods.Poisson.poisson_helpers import *


# default update
def unsplit_update(U, params, boundaries, update_coeffs, f_all):
    U_old = U.copy()
    U_new = U.copy()

    Nx, Ny, dx, dy, omega = params.Nx, params.Ny, params.dx, params.dy, params.omega
    c1, c2 = update_coeffs
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je

    # 2nd order spatial on LHS
    # for i in range(ib, ie + 1):
    #     for j in range(jb, je + 1):
    #         # if (i + j) % 2 == parity[1]:
    #         # edge pts need 1st order stencil
    #         # if i in [ib, ie] or j in [jb, je]:
    #         U_new[i, j] = (U[i, j] + c1 * ((dy * (U_new[i-1, j] - U[i-1, j]))
    #                        + (dx * (U_new[i, j-1] - U[i, j-1]))
    #                        + (dt * (isotropic_9pt(U, i, j, dx) - f_all[i, j]))))
    #         # else:
    #         #     U_new[i, j] = (U[i, j] + (3.0 * c1 * ((dy * (4.0 * U_new[i - 1, j] - U_new[i - 2, j]
    #         #                                                  - 4.0 * U[i - 1, j] + U[i - 2, j]))
    #         #                                           + (dx * (4.0 * U_new[i, j - 1] - U_new[i, j - 2]
    #         #                                                    - 4.0 * U[i, j - 1] + U[i, j - 2]))
    #         #                                           + (2.0 * dt * (isotropic_9pt(U, i, j, dx) - f_all[i, j])))))
    #
    #         # SOR update
    #         U_new[i, j] = U_old[i, j] + (omega * (U_new[i, j] - U_old[i, j]))
    #
    # using centered differences
    # for i in range(ib, ie + 1):
    #     for j in range(jb, je + 1):
    #         U_new[i, j] = (c1 * ((dy * (U_new[i-1, j] + 0.5 * (U[i+1, j] - U[i-1, j])))
    #                        + (dx * (U_new[i, j-1] + 0.5 * (U[i, j+1] - U[i, j-1])))
    #                        + (c2 * (centered_2D_laplacian(U, i, j, params) - f_all[i, j]))))
    #
    #         # SOR update
    #         U_new[i, j] = U_old[i, j] + (omega * (U_new[i, j] - U_old[i, j]))

    for i in range(ib, ie + 1):
        for j in range(jb, je + 1):
            U_new[i, j] = (U[i, j] + c1 * ((dy * (U_new[i - 1, j] - U[i - 1, j]))
                                           + (dx * (U_new[i, j - 1] - U[i, j - 1]))
                                           + (c2 * (centered_2D_laplacian(U, i, j, params) - f_all[i, j]))))

            # SOR update
            U_new[i, j] = U_old[i, j] + (omega * (U_new[i, j] - U_old[i, j]))

    # update BCs
    bc.apply_bcs_poisson_2d(U_new, params, boundaries)

    return U_new


def forward_x_back_y(U, params, boundaries, update_coeffs, f_all):
    U_old = U.copy()
    U_new = U.copy()

    bxdy, bydx, c1, c2 = update_coeffs
    omega, ib, ie, jb, je = params.omega, params.ib, params.ie, params.jb, params.je
    den = -bxdy + bydx

    # forward in x, backward in y
    for i in range(ie, ib-1, -1):
        for j in range(jb, je+1):
            U_new[i, j] = U[i, j] + (
                    bxdy * (U_new[i + 1, j] - U[i + 1, j])
                    - bydx * (U_new[i, j - 1] - U[i, j - 1])
                    + c2 * (centered_2D_laplacian(U, i, j, params) - f_all[i, j])
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
    for i in range(ib, ie+1):
        for j in range(je, jb-1, -1):
            U_new[i, j] = U[i, j] + (
                    -bxdy * (U_new[i - 1, j] - U[i - 1, j])
                    + bydx * (U_new[i, j + 1] - U[i, j + 1])
                    + c2 * (centered_2D_laplacian(U, i, j, params) - f_all[i, j])
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
    den = bxdy + bydx   # we will need to negate this for the update (both terms)

    # forward in both
    for i in range(ie, ib-1, -1):
        for j in range(je, jb-1, -1):
            U_new[i, j] = U[i, j] + (
                    bxdy * (U_new[i + 1, j] - U[i + 1, j])
                    + bydx * (U_new[i, j + 1] - U[i, j + 1])
                    - c2 * (centered_2D_laplacian(U, i, j, params) - f_all[i, j])
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
                                           + (c2 * (centered_2D_laplacian(U, i, j, params) - f_all[i, j]))))

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
        i_range = range(ib, ie + 1)          # backward-dominant in x
    else:
        i_range = range(ie, ib - 1, -1)      # forward-dominant in x

    if beta >= 0.5:
        j_range = range(jb, je + 1)          # backward-dominant in y
    else:
        j_range = range(je, jb - 1, -1)      # forward-dominant in y

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

            rhs = c2 * (centered_2D_laplacian(U, i, j, params) - f_all[i, j])

            U_new[i, j] = U[i, j] + (rhs - bxdy * x_inc - bydx * y_inc) / den

            # SOR update
            U_new[i, j] = U_old[i, j] + omega * (U_new[i, j] - U_old[i, j])

    bc.apply_bcs_poisson_2d(U_new, params, boundaries)

    return U_new


# in progress
def back_x_back_y_2nd(U, params, boundaries, update_coeffs, f_all):
    U_old = U.copy()
    U_new = U.copy()

    # bxdy, bydx, c1, c2 = update_coeffs
    omega, ib, ie, jb, je = params.omega, params.ib, params.ie, params.jb, params.je

    for i in range(ib, ie + 1):
        for j in range(jb, je + 1):
            # ...

            # SOR update
            U_new[i, j] = U_old[i, j] + (omega * (U_new[i, j] - U_old[i, j]))

    # update BCs
    bc.apply_bcs_poisson_2d(U_new, params, boundaries)

    return U_new


def get_params(config):
    # get parameters from config class
    cfl, dx, dy = config.cfl, config.dx, config.dy

    # pseudo-time step
    dt_factor = 1.00
    dt = dt_factor * cfl * min(dx, dy)

    # default-method coefficients
    c1 = 1.0 / (dx + dy)
    c2 = dt * dx * dy

    # ------------------------------------------------------------
    # theta / direction choice
    # ------------------------------------------------------------
    random = True

    # examples:
    # theta = 0.5105761553861832
    # theta = 1.1355425113197994
    # theta = 1.1233512818763773
    # theta = 4.162255099171517

    ### best (up to date) examples ###
    # theta = 1.061741801784727     # omega 0.910 -> 65 iterations
    # theta = 0.25 * np.pi      # default: omega 0.913 -> 67 iterations
    theta = 0.25 * np.pi    # backward

    # parabolic (explicit) best is 52 iterations, about 1.5x faster

    # randomized
    # theta = sample_theta_with_denominator_buffer(
    #     dx, dy,
    #     angle_buffer=np.pi / 12,
    #     dmin=1e-6
    # )

    # examples:
    # alpha = beta = 0.80   # mostly backward
    # alpha = beta = 0.20   # mostly forward
    alpha = 0.8
    beta = 0.8

    # sx = 1.0 if alpha > 0.5 else -1.0
    # sy = 1.0 if beta > 0.5 else -1.0
    #
    # bx = sx * np.sqrt(2)
    # by = sy * np.sqrt(2)

    bx, by = np.cos(theta), np.sin(theta)
    bx *= np.sqrt(2)    # normalization
    by *= np.sqrt(2)

    bxdy = bx * dy
    bydx = by * dx
    c1r = 1.0 / (bxdy + bydx)
    update_coeffs = (bxdy, bydx, c1r, c2)

    # weighted forward/backward hybrid parameters
    eps_alpha = 0.05      # keep alpha,beta away from 0.5
    use_weighted_hybrid = True


    if abs(alpha - 0.5) < eps_alpha or abs(beta - 0.5) < eps_alpha:
        raise ValueError("alpha and beta must stay away from 0.5 by at least eps_alpha")

    params = {
        "dt": dt,
        "spatial_order": 1,     # order of LHS spatial discretization
        "random": random,
        "theta": theta,
        "bx": bx,
        "by": by,
        "update_coeffs": update_coeffs,
        "c1": c1,
        "use_weighted_hybrid": use_weighted_hybrid,
        "alpha": alpha,
        "beta": beta,
        "eps_alpha": eps_alpha,
    }

    # print(f"theta = {theta} | bx = {bx} | by = {by}")

    return params


def full_sweep(U, params, boundaries, config):
    f_all = config.f_all
    update_coeffs = params["update_coeffs"]
    _, _, _, c2 = update_coeffs
    use_weighted_hybrid = params.get("use_weighted_hybrid", False)
    alpha = params.get("alpha", 1.0)
    beta = params.get("beta", 1.0)

    if use_weighted_hybrid:
        U = weighted_hybrid(U, config, boundaries, update_coeffs, f_all, alpha, beta)
    elif params["random"]:
        theta_mod = params["theta"] % (2.0 * np.pi)

        if 0.0 <= theta_mod < 0.5 * np.pi:
            # backward in both
            U = back_x_back_y(U, config, boundaries, update_coeffs, f_all)
        elif 0.5 * np.pi <= theta_mod < np.pi:
            # forward in x, backward in y
            U = forward_x_back_y(U, config, boundaries, update_coeffs, f_all)
        elif np.pi <= theta_mod < 1.5 * np.pi:
            # forward in both
            U = forward_x_forward_y(U, config, boundaries, update_coeffs, f_all)
        else:
            # backward in x, forward in y
            U = back_x_forward_y(U, config, boundaries, update_coeffs, f_all)
    else:
        U = unsplit_update(U, config, boundaries, (params["c1"], c2), f_all)

    return U
