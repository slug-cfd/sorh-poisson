import numpy as np

def centered_2D_laplacian(U, i, j, config):
    dx, dy = config.dx, config.dy
    x_disc = (1 / dx**2) * (U[i+1, j] - (2.0 * U[i, j]) + U[i-1, j])
    y_disc = (1 / dy**2) * (U[i, j+1] - (2.0 * U[i, j]) + U[i, j-1])
    return x_disc + y_disc


def get_all_laplace(U, params) -> np.ndarray:
    # x, y = params.x, params.y
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
    laplace_array = np.zeros_like(U)

    for i in range(ib, ie+1):
        for j in range(jb, je+1):
            laplace_array[i, j] = centered_2D_laplacian(U, i, j, params)

    return laplace_array


def isotropic_9pt(U, i, j, h):
    c = 1.0 / (6.0 * h**2)
    update = (4.0 * (U[i+1, j] + U[i-1, j] + U[i, j+1] + U[i, j-1])
              + (U[i+1, j+1] + U[i+1, j-1] + U[i-1, j+1] + U[i-1, j-1])
              - (20.0 * U[i, j]))
    return c * update

# helper function: build a tridiagonal matrix
def tridiag(n, a1, a0, b1):
    """
    Build an n×n tridiagonal matrix with
    1st subdiag = a1
    main diag   = a0
    1st super   = b1
    """
    A = np.zeros((n, n))
    i = np.arange(n)

    A[i, i] = a0                     # main
    A[i[1:], i[:-1]] = a1            # sub1
    A[i[:-1], i[1:]] = b1            # super1

    return A
