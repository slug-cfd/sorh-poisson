import numpy as np

def sample_theta_buffered(buffer_angle=0.1, rng=None):
    """
    Sample theta in [0, 2pi), staying buffer_angle away from
    0, pi/2, pi, 3pi/2, 2pi.

    Parameters
    ----------
    buffer_angle : float
        Angular buffer in radians. Must satisfy 0 < buffer_angle < pi/4.
    rng : np.random.Generator or None
        Optional random number generator.

    Returns
    -------
    theta : float
        Sampled angle in [0, 2pi).
    """
    if not (0.0 < buffer_angle < 0.25 * np.pi):
        raise ValueError("buffer_angle must satisfy 0 < buffer_angle < pi/4")

    if rng is None:
        rng = np.random.default_rng()

    quadrant_width = 0.5 * np.pi - 2.0 * buffer_angle
    # q = rng.integers(0, 4)  # choose one of 4 quadrants
    q = 0
    theta = q * (0.5 * np.pi) + buffer_angle + quadrant_width * rng.random()

    return theta


def sample_theta_with_denominator_buffer(dx, dy, angle_buffer=0.1, dmin=1e-8, rng=None):
    """
    Sample theta away from axis directions and also avoid small |bx*dy + by*dx|.
    """
    if rng is None:
        rng = np.random.default_rng()

    for _ in range(1000):
        theta = sample_theta_buffered(buffer_angle=angle_buffer, rng=rng)
        bx = np.cos(theta)
        by = np.sin(theta)
        bx *= np.sqrt(2)
        by *= np.sqrt(2)
        D = bx * dy + by * dx
        if abs(D) > dmin:
            return theta

    raise RuntimeError("Failed to sample a valid theta after many attempts.")


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
