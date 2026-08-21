from methods.poisson.poisson_helpers import *
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def get_params(config):
    if config.laplacian_stencil != "centered":
        raise NotImplementedError(
            "p_ADI currently supports only laplacian_stencil='centered'."
        )
    if config.ngc != 1:
        raise NotImplementedError("p_ADI currently supports only ghost_layers=1.")

    # # get parameters from config class
    cfl, dx, dy, Nx, Ny = config.cfl, config.dx, config.dy, config.Nx, config.Ny

    # experiment on dt
    dt_factor = 1.0     # best: 32.0
    dt = dt_factor * cfl * np.min([dx, dy])

    # set up matrices for implicit-explicit systems
    # Sparse identities (CSR by default)
    Ix = sp.identity(Nx, format="csr")
    Iy = sp.identity(Ny, format="csr")

    # Sparse tridiagonals
    Tx = sp.diags([1.0, -2.0, 1.0], offsets=[-1, 0, 1], shape=(Nx, Nx), format="csr")
    Ty = sp.diags([1.0, -2.0, 1.0], offsets=[-1, 0, 1], shape=(Ny, Ny), format="csr")

    cx = (0.5 * dt / dx ** 2)
    cy = (0.5 * dt / dy ** 2)

    Dxx = cx * Tx
    Dyy = cy * Ty

    Ixm = Ix - Dxx  # still CSR
    Iym = Iy - Dyy

    solve_x = spla.factorized(Ixm.tocsc())
    solve_y = spla.factorized(Iym.tocsc())

    params = {
        "dt": dt,
        "solve_x": solve_x,
        "solve_y": solve_y,
    }

    return params


def full_sweep(U, params, boundaries, config):
    Nx, Ny, dx, dy = config.Nx, config.Ny, config.dx, config.dy
    ib, ie, jb, je = config.ib, config.ie, config.jb, config.je
    omega, f_all = config.omega, config.f_all
    dt, solve_x, solve_y = params["dt"], params["solve_x"], params["solve_y"]

    phi_xL, phi_xR, phi_yB, phi_yT = [side[0] for side in boundaries]
    U_old = U.copy()

    # 1st step (implicit in x, explicit in y)
    U_new = U.copy()

    for j in range(jb, je+1):
        f_j = f_all[ib:ie+1, j]
        h_j = np.zeros(Nx)
        h_j[0] += phi_xL[j] / (dx ** 2)
        h_j[-1] += phi_xR[j] / (dx ** 2)
        Dyy_j = (U[ib:ie+1, j-1] - (2.0 * U[ib:ie+1, j]) + U[ib:ie+1, j+1]) / dy**2
        rhs = U[ib:ie+1, j] + (0.5 * dt * (Dyy_j - f_j + h_j))
        U_new[ib:ie+1, j] = solve_x(rhs)

    # 2nd step (implicit in y, explicit in x)
    U = U_new.copy()

    for i in range(ib, ie+1):
        f_i = f_all[i, jb:je+1]
        h_i = np.zeros(Ny)
        h_i[0] += phi_yB[i] / (dy ** 2)
        h_i[-1] += phi_yT[i] / (dy ** 2)
        Dxx_i = (U[i-1, jb:je+1] - (2.0 * U[i, jb:je+1]) + U[i+1, jb:je+1]) / dx**2
        rhs = U[i, jb:je+1] + (0.5 * dt * (Dxx_i - f_i + h_i))
        U_new[i, jb:je+1] = solve_y(rhs)

    # global SOR update
    U_new = ((1.0 - omega) * U_old) + (omega * U_new)

    # BC updates are built into the implicit solves
    # so no explicit BC update is needed here

    return U_new
