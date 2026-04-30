import BC_2D as bc
from methods.Poisson.poisson_helpers import *

# default update
def unsplit_update(U, params, boundaries, update_coeffs, f_all):
    U_old = U.copy()
    U_new = U.copy()

    Nx, Ny, dx, dy, omega = params.Nx, params.Ny, params.dx, params.dy, params.omega
    c1, c2 = update_coeffs
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je

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
                    -bxdy * (U_new[i + 1, j] - U[i + 1, j])
                    + bydx * (U_new[i, j - 1] - U[i, j - 1])
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
                    bxdy * (U_new[i - 1, j] - U[i - 1, j])
                    - bydx * (U_new[i, j + 1] - U[i, j + 1])
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
