def parabolic(params, phi_exact, get_all_f):
    # get parameters from config class
    cfl, dx, dy, Nx, Ny = params.cfl, params.dx, params.dy, params.Nx, params.Ny
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
    omega, iter_max = params.omega, params.iter_max
    norm = params.norm

    boundaries = bc.get_bcs_poisson_2d(params, phi_exact)

    U = params.U0.copy()
    bc.apply_bcs_poisson_2d(U, params, boundaries)

    # apply f(x, y) over the whole 2D grid
    f_all = get_all_f(params)

    # dt = cfl * ((dx ** 2) / (2*v))  # 1D stability condition

    dt = cfl / (2 * (1 / dx ** 2 + 1 / dy ** 2))
    # equivalently: dt = cfl * (dx**2 * dy**2) / (2*v*(dx**2 + dy**2))

    # const. factors for method update rule
    # rx = dt / (dx ** 2)
    # ry = dt / (dy ** 2)
    # midpoint = 1.0 - (2.0 * rx) - (2.0 * ry)

    # dt *= 4.20
    # dt_mod = dt * 4.052

    #  error array (one value per iteration)
    err_conv = np.zeros(iter_max)

    t = 0.0
    iter_count = 0

    threshold = 1e-3  # error/residual tolerance
    res_max = 1e7  # upper bound to avoid blow-up in unstable cases
    res_diff = 1e2  # difference in norm of the solution between current and previous iteration

    # we need a separate variable for res_exact because we are still
    # using res_diff as the criterion to exit the iteration loop,
    # even though we are returning and plotting res_exact.
    res_exact = 1e4
    res_prev = 0.0

    metric = "exact"

    x, y, rho0, R, G = params.x, params.y, params.rho0, params.R, params.G
    X, Y = np.meshgrid(x, y, indexing='ij')

    # print("using exact solution")
    U_exact = phi_exact(X, Y, rho0, R, G)

    while res_diff > threshold and iter_count < iter_max and res_exact < res_max:
        #  store the old solution to measure the steadiness
        U_old = U.copy()

        # Lexicographic GS - simple sequential updates
        # for i in range(ib, ie + 1):
        #     for j in range(jb, je + 1):
        #         U[i, j] = (midpoint * U[i, j]
        #                    + rx * (U[i - 1, j] + U[i + 1, j])
        #                    + ry * (U[i, j - 1] + U[i, j + 1]) - (dt * f_all[i, j]))
        #
        #         # SOR update
        #         U[i, j] = U_old[i, j] + (omega * (U[i, j] - U_old[i, j]))

        # Red-Black GS - checkerboard update pattern
        # -> helps to remove direction dependence and asymmetry issues

        # # RED sweep
        # for i in range(ib, ie + 1):
        #     for j in range(jb, je + 1):
        #         if (i + j) % 2 == 0:
        #             U[i, j] = midpoint * U[i, j] \
        #                       + rx * (U[i - 1, j] + U[i + 1, j]) \
        #                       + ry * (U[i, j - 1] + U[i, j + 1]) - (dt_mod * f_all[i, j])
        #
        #             # SOR update
        #             U[i, j] = U_old[i, j] + (omega * (U[i, j] - U_old[i, j]))
        #
        # # BLACK sweep
        # for i in range(ib, ie + 1):
        #     for j in range(jb, je + 1):
        #         if (i + j) % 2 == 1:
        #             U[i, j] = midpoint * U[i, j] \
        #                       + rx * (U[i - 1, j] + U[i + 1, j]) \
        #                       + ry * (U[i, j - 1] + U[i, j + 1]) - (dt_mod * f_all[i, j])
        #
        #             # SOR update
        #             U[i, j] = U_old[i, j] + (omega * (U[i, j] - U_old[i, j]))

        # Red-Black GS - checkerboard update pattern
        # -> helps to remove direction dependence and asymmetry issues
        # sweep over INTERNAL indices only: [ib, ie]

        # RED sweep
        for i in range(ib, ie + 1):
            for j in range(jb, je + 1):
                if (i + j) % 2 == 0:
                    U[i, j] = U[i, j] + (dt * (centered_2D_laplacian(U, i, j, params) - f_all[i, j]))

                    # SOR update
                    U[i, j] = U_old[i, j] + (omega * (U[i, j] - U_old[i, j]))

        # BLACK sweep
        for i in range(ib, ie + 1):
            for j in range(jb, je + 1):
                if (i + j) % 2 == 1:
                    U[i, j] = U[i, j] + (dt * (centered_2D_laplacian(U, i, j, params) - f_all[i, j]))

                    # SOR update
                    U[i, j] = U_old[i, j] + (omega * (U[i, j] - U_old[i, j]))

        # update BCs
        bc.apply_bcs_poisson_2d(U, params, boundaries)

        # difference between laplacian(U) and f
        # this can also be used as the convergence condition, but is more costly
        # l_all = get_all_laplace(U, params)
        # err_diff = l_all - f_all

        # difference between subsequent iterations
        err_diff = U - U_old

        if metric == "exact":
            err_exact = U - U_exact
            res_exact = np.linalg.norm(err_exact[ib:ie + 1, jb:je + 1], ord=norm)
            res_diff = np.abs(res_exact - res_prev)
            res_prev = res_exact
        else:
            # take the norm and average
            res_diff = np.linalg.norm(err_diff[ib:ie + 1, jb:je + 1], ord=norm)
            if norm in [1, 2, "fro"]:
                res_diff /= np.sqrt(Nx * Ny)

        err_conv[iter_count] = res_exact

        #  update clock and iteration count
        t = t + dt
        iter_count += 1

    # return U, t, iter_count, res, err_conv, err_exact
    return U, t, iter_count, res_exact, err_conv


def unsplit(params, phi_exact, get_all_f):
    # get parameters from config class
    cfl, dx, dy, Nx, Ny = params.cfl, params.dx, params.dy, params.Nx, params.Ny
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
    norm, iter_max = params.norm, params.iter_max

    # convergence history: store exact error at each iteration
    err_conv = np.zeros(iter_max)

    t = 0.0
    iter_count = 0

    # stopping / safety parameters
    threshold = 1e-7          # stopping tolerance for update norm
    res_max = 1e7             # blow-up guard
    res_update = 1e2
    res_exact = 1e4

    boundaries = bc.get_bcs_poisson_2d(params, phi_exact)

    U = params.U0.copy()
    bc.apply_bcs_poisson_2d(U, params, boundaries)

    # apply f(x, y) over the whole 2D grid
    f_all = get_all_f(params)

    # pseudo-time step
    dt_factor = 1.0
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
    theta = 4.162255099171517

    # theta = sample_theta_with_denominator_buffer(
    #     dx, dy,
    #     angle_buffer=np.pi / 12,
    #     dmin=1e-6
    # )

    bx, by = np.cos(theta), np.sin(theta)
    bx *= np.sqrt(2)
    by *= np.sqrt(2)

    bxdy = bx * dy
    bydx = by * dx
    c1r = 1.0 / (bxdy + bydx)
    update_coeffs = (bxdy, bydx, c1r, c2)

    # exact solution on full grid
    x, y, rho0, R, G = params.x, params.y, params.rho0, params.R, params.G
    X, Y = np.meshgrid(x, y, indexing='ij')
    U_exact = phi_exact(X, Y, rho0, R, G)

    # track best exact-error iterate separately
    best_exact = np.inf
    U_best = U.copy()

    # ------------------------------------------------------------
    # main iteration loop
    # ------------------------------------------------------------
    while iter_count < iter_max:
        U_old = U.copy()

        if random:
            theta_mod = theta % (2.0 * np.pi)

            if 0.0 <= theta_mod < 0.5 * np.pi:
                # backward in both
                U = back_x_back_y(U, params, boundaries, update_coeffs, f_all)
            elif 0.5 * np.pi <= theta_mod < np.pi:
                # forward in x, backward in y
                U = forward_x_back_y(U, params, boundaries, update_coeffs, f_all)
            elif np.pi <= theta_mod < 1.5 * np.pi:
                # forward in both
                U = forward_x_forward_y(U, params, boundaries, update_coeffs, f_all)
            else:
                # backward in x, forward in y
                U = back_x_forward_y(U, params, boundaries, update_coeffs, f_all)
        else:
            U = unsplit_update(U, params, boundaries, (c1, c2), f_all)

        # --------------------------------------------------------
        # 1) update norm: used for stopping
        # --------------------------------------------------------
        err_update = U - U_old
        res_update = np.linalg.norm(err_update[ib:ie + 1, jb:je + 1], ord=norm)
        if norm in [1, 2, "fro"]:
            res_update /= np.sqrt(Nx * Ny)

        # --------------------------------------------------------
        # 2) exact error: diagnostics / plotting only
        # --------------------------------------------------------
        err_exact = U - U_exact
        res_exact = np.linalg.norm(err_exact[ib:ie + 1, jb:je + 1], ord=norm)
        if norm in [1, 2, "fro"]:
            res_exact /= np.sqrt(Nx * Ny)

        # use exact error for residual history
        err_conv[iter_count] = res_exact

        # store best exact iterate seen so far
        if res_exact < best_exact:
            best_exact = res_exact
            U_best = U.copy()

        # update time / iteration count
        t += dt
        iter_count += 1

        # blow-up guard
        if (not np.isfinite(res_update)) or (not np.isfinite(res_exact)):
            break
        if res_update > res_max or res_exact > res_max:
            break

        # stopping criterion: discrete iteration has converged
        if res_update <= threshold:
            break

    # ------------------------------------------------------------
    # Return the best-exact iterate, but keep the reported
    # "final exact residual" consistent with that returned U.
    # ------------------------------------------------------------
    U = U_best
    err_exact = U - U_exact
    res_exact = np.linalg.norm(err_exact[ib:ie + 1, jb:je + 1], ord=norm)
    if norm in [1, 2, "fro"]:
        res_exact /= np.sqrt(Nx * Ny)

    if random and res_exact < 1.0 and iter_count < 1000:
        print(f"theta = {theta} | bx = {bx} | by = {by}")

    return U, t, iter_count, res_exact, err_conv


# def unsplit(params, phi_exact, get_all_f):
#     # get parameters from config class
#     cfl, dx, dy, Nx, Ny = params.cfl, params.dx, params.dy, params.Nx, params.Ny
#     ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
#     norm, iter_max = params.norm, params.iter_max
#
#     # error array (one value per iteration)
#     err_conv = np.zeros(iter_max)
#
#     t = 0.0
#     iter_count = 0
#
#     threshold = 1e-3  # error/residual tolerance
#     res_max = 1e7  # upper bound to avoid blow-up in unstable cases
#     res_diff = 1e2  # difference in norm of the solution between current and previous iteration
#
#     # we need a separate variable for res_exact because we are still
#     # using res_diff as the criterion to exit the iteration loop,
#     # even though we are returning and plotting res_exact.
#     res_exact = 1e4
#     res_prev = 0.0
#
#     boundaries = bc.get_bcs_poisson_2d(params, phi_exact)
#
#     U = params.U0.copy()
#     bc.apply_bcs_poisson_2d(U, params, boundaries)
#
#     # apply f(x, y) over the whole 2D grid
#     f_all = get_all_f(params)
#
#     # these parameters are always constant, so pre-calculate them
#     dt_factor = 1.0
#     dt = dt_factor * cfl * np.min([dx, dy])
#
#     # original parameters
#     c1 = 1.0 / (dx + dy)
#     c2 = dt * dx * dy
#
#     # random parameters
#     # a = 0.1
#     # b = 2.0 * np.pi - 0.1
#
#     # theta = 0.5105761553861832    # 84 iterations
#     # theta = 1.1355425113197994      # 75 iterations
#
#     # theta = 1.1233512818763773      # 77 iterations
#
#     theta = 4.162255099171517
#
#     # theta = 0.5 * np.pi * np.random.random()
#     # theta = 7.0 * np.pi / 4.0
#
#     # theta = (b - a) * np.random.random() + a
#     # theta = sample_theta_with_denominator_buffer(
#     #     dx, dy,
#     #     angle_buffer=np.pi / 12,
#     #     dmin=1e-6
#     # )
#
#     bx, by = np.cos(theta), np.sin(theta)
#     bx *= np.sqrt(2)
#     by *= np.sqrt(2)
#
#     # print(f"theta = {theta} | bx = {bx} | by = {by}")
#
#     bxdy = bx * dy
#     bydx = by * dx
#     c1r = 1.0 / (bxdy + bydx)
#     update_coeffs = (bxdy, bydx, c1r, c2)
#
#     random = True
#
#     x, y, rho0, R, G = params.x, params.y, params.rho0, params.R, params.G
#     X, Y = np.meshgrid(x, y, indexing='ij')
#
#     U_exact = phi_exact(X, Y, rho0, R, G)  # exact solution over full grid
#
#     while res_diff > threshold and iter_count < iter_max and res_exact < res_max:
#         U_old = U.copy()
#
#         if random:
#             if theta < 0.5 * np.pi:     # backward in both
#                 # print("b/b")
#                 U = back_x_back_y(U, params, boundaries, update_coeffs, f_all)
#             elif 0.5 * np.pi < theta < np.pi:       # forward in x, backward in y
#                 # print("f/b")
#                 U = forward_x_back_y(U, params, boundaries, update_coeffs, f_all)
#             elif np.pi < theta < 1.5 * np.pi:       # forward in both
#                 # print("f/f")
#                 U = forward_x_forward_y(U, params, boundaries, update_coeffs, f_all)
#             else:                       # backward in x, forward in y
#                 # print("b/f")
#                 U = back_x_forward_y(U, params, boundaries, update_coeffs, f_all)
#         else:   # default
#             update_coeffs = (c1, c2)
#             U = unsplit_update(U, params, boundaries, update_coeffs, f_all)
#
#         # difference between laplacian(U) and f
#         # this can also be used as the convergence condition, but is more costly
#         # l_all = get_all_laplace(U, params)
#         # err_diff = l_all - f_all
#
#         # difference between subsequent iterations
#         err_diff = U - U_old
#         res_update = np.linalg.norm(err_diff[ib:ie + 1, jb:je + 1], ord=norm)
#         if norm in [1, 2, "fro"]:
#             res_update /= np.sqrt(Nx * Ny)
#
#         # exact error for diagnostics/plotting only
#         err_exact = U - U_exact
#         res_exact = np.linalg.norm(err_exact[ib:ie + 1, jb:je + 1], ord=norm)
#         if norm in [1, 2, "fro"]:
#             res_exact /= np.sqrt(Nx * Ny)
#
#         res_diff = res_update
#         err_conv[iter_count] = res_exact
#
#         #  update clock and iteration count
#         t = t + dt
#         iter_count += 1
#
#     if random and res_exact < 1.0 and iter_count < 1000:
#         print(f"theta = {theta} | bx = {bx} | by = {by}")
#
#     return U, t, iter_count, res_exact, err_conv
