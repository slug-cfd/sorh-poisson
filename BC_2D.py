import numpy as np

def update_BCs(U, params, phi_exact=None):
    a, b, c, d = 1.0, 2.0, 1.0, 2.0

    # update BCs
    match params.bc_type:
        case "poisson":
            apply_bcs_poisson_2d(U, params, phi_exact)
        case "linear_in_x":
            # BC: U(x, y) = x
            linear_in_x(U, params.x)
        case "other_1":
            # BC: spatial oscillations
            coeffs = [a, b, c, d]
            sinusoidal(U, params.x, coeffs)
        case "other_2":
            # BC: combination (mixed nonlinear functions)
            coeffs = [a, b]
            combination(U, params.x, coeffs)
        case _:
            print("Unrecognized bc_type: ", params.bc_type)


def get_bcs_poisson_2d(params, phi_exact):
    """
    U: (Nx+2, Ny+2) with 1 ghost layer each side.
    x, y: arrays of length Nx+2, Ny+2 (include interior and ghost cells)
    Reflection (Neumann 0): x=0, y=0
    Dirichlet: x=20R, y=20R using phi_exact
    """
    x, y, rho0, R, G = (params.x, params.y,
                        params.rho0, params.R, params.G)
    # Nx, Ny, dx, dy = (params.Nx, params.Ny, params.dx, params.dy)
    # ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
    xa, xb, ya, yb = params.xa, params.xb, params.ya, params.yb

    # # Ghost cells: all Dirichlet
    # U[0, :] = phi_exact(xa-dx, y, rho0, R, G)  # x = xa (left)
    # U[ie+1, :] = phi_exact(xb+dx, y, rho0, R, G)  # x = xb (right)
    #
    # U[1:-1, 0] = phi_exact(x[1:-1], ya-dy, rho0, R, G)  # y = ya (bottom)
    # U[1:-1, je+1] = phi_exact(x[1:-1], yb+dy, rho0, R, G)   # y = yb (top)
    #
    # # Physical boundaries: all Dirichlet
    # U[1, 1:-1] = phi_exact(xa, y[1:-1], rho0, R, G)  # x = xa (left)
    # U[ie, 1:-1] = phi_exact(xb, y[1:-1], rho0, R, G)  # x = xb (right)
    #
    # U[2:-2, 1] = phi_exact(x[2:-2], ya, rho0, R, G)  # y = ya (bottom)
    # U[2:-2, je] = phi_exact(x[2:-2], yb, rho0, R, G)   # y = yb (top)

    # --- Reflection at x=0 (left): ghost mirrors first physical column
    # U[0, jb:je+1] = U[1, jb:je+1]
    #
    # # try Dirichlet
    # # U[0, :] = phi_exact(xa, y, rho0, R, G)  # x = xa (left)
    # # U[:-1, 0] = phi_exact(x, ya, rho0, R, G)  # y = ya (bottom)
    #
    # # --- Reflection at y=0 (bottom): ghost mirrors first physical row
    # U[ib:ie+1, 0] = U[ib:ie+1, 1]
    #
    # # --- Dirichlet on physical boundary nodes ---
    # U[ie, :] = phi_exact(xb, y, rho0, R, G)     # x = xb (right)
    # U[:-1, je] = phi_exact(x[:-1], yb, rho0, R, G)   # y = yb (top)
    #
    # # --- Ghosts adjacent to Dirichlet (copy physical boundary) ---
    # U[ie+1, :] = U[ie, :]
    # U[:-1, je+1] = U[:-1, je+1]
    #
    # # Corners
    # U[0, 0]       = U[1, 1]
    # U[0, je+1]    = U[1, je]
    # U[je+1, 0]    = U[je, 1]

    # xa, xb, ya, yb = config.xa, config.xb, config.ya, config.yb
    # x, y = config.x, config.y

    # ---- Enforce U(x, y) = phi_exact(x, y) on the physical boundary ----
    phi_xL = phi_exact(xa, y, rho0, R, G)
    phi_xR = phi_exact(xb, y, rho0, R, G)

    phi_yB = phi_exact(x, ya, rho0, R, G)
    phi_yT = phi_exact(x, yb, rho0, R, G)

    BCs = [phi_xL, phi_xR, phi_yB, phi_yT]

    return BCs

def apply_bcs_poisson_2d(U, params, boundaries):
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
    phi_xL, phi_xR, phi_yB, phi_yT = boundaries

    # Apply to ghost cells
    U[0, :] = phi_xL
    U[ie + 1, :] = phi_xR
    U[:, 0] = phi_yB
    U[:, je + 1] = phi_yT


# solution profile: U(x, y) = x
def linear_in_x(U, x):
    # x=0 -> u=0
    U[0, :] = 0.0
    # x=1 -> u=1
    U[-1, :] = 1.0
    # y=0 -> u=x
    U[:, 0] = x
    # y=1 -> u=x
    U[:, -1] = x

    return U

def sinusoidal(U, x, params):
    a, b, c, d = params

    U[:, 0] = a * np.sin(b * np.pi * x)
    U[:, -1] = 0.0
    U[0, :] = c * np.sin(d * np.pi * x)
    U[-1, :] = 0.0

    return U


def combination(U, x, params):
    a, b = params

    U[:, 0] = a * np.sin(b * np.pi * x)
    U[:, -1] = 0.0
    U[0, :] = -np.sqrt(x)
    U[-1, :] = np.sqrt(x)

    return U
