import numpy as np

from .types import PoissonTestProblem


# Table 3: Polar Star Poisson Problem Parameters
POLAR_STARS = (
    {"x0": -0.5, "y0": -0.5, "r0": 0.2, "r1": 0.3, "n": 3},
    {"x0": 0.5, "y0": -0.5, "r0": 0.3, "r1": 0.4, "n": 4},
    {"x0": 0.0, "y0": 0.5, "r0": 0.4, "r1": 0.5, "n": 5},
)


def sech(z):
    return 1.0 / np.cosh(z)


def radius(x, y, x0, y0):
    return np.sqrt((x - x0) ** 2 + (y - y0) ** 2)


def theta_polar_star(x, y, x0, y0):
    return np.arctan2(y - y0, x - x0)


def phi(x, y, star, epsilon):
    rad = radius(x, y, star["x0"], star["y0"])
    ang = theta_polar_star(x, y, star["x0"], star["y0"])

    return (
        rad
        - star["r0"] * (star["r1"] * np.cos(star["n"] * ang) + 1.0)
    ) / epsilon


def p(x, y, star):
    ang = theta_polar_star(x, y, star["x0"], star["y0"])
    return star["n"] * star["r0"] * star["r1"] * np.sin(star["n"] * ang)


def exact(x, y, config):
    epsilon = config.epsilon

    U = np.zeros_like(np.asarray(x) + np.asarray(y), dtype=float)

    for star in POLAR_STARS:
        phi_i = phi(x, y, star, epsilon)
        U += 0.5 * (1.0 - np.tanh(phi_i))

    return U


def rhs(x, y, config):
    # BUG FIX 1: use the same epsilon as `exact` so RHS and exact solution
    # are consistent.  Previously this was hardcoded to 1.0 while `exact`
    # read epsilon from config, making them describe *different* functions.
    epsilon = config.epsilon

    F = np.zeros_like(np.asarray(x) + np.asarray(y), dtype=float)

    for star in POLAR_STARS:
        rad = radius(x, y, star["x0"], star["y0"])
        rad_safe = np.where(rad > 1e-12, rad, np.inf)

        ang = theta_polar_star(x, y, star["x0"], star["y0"])
        phi_i = phi(x, y, star, epsilon)
        p_i = p(x, y, star)

        sech2_phi = sech(phi_i) ** 2
        tanh_phi = np.tanh(phi_i)

        n = star["n"]
        r0 = star["r0"]
        r1 = star["r1"]

        # s1 through s4 match the paper's definitions (eq. 4.4 and below it).
        # s2 already carries its own minus sign per the paper's definition.

        s1_i = (p_i ** 2) * tanh_phi * sech2_phi / (epsilon ** 2)

        s2_i = (
            -(n ** 2) * r0 * r1
            * np.cos(n * ang)
            * sech2_phi
            / (2.0 * epsilon)
        )

        s3_i = tanh_phi * sech2_phi / (epsilon ** 2)

        s4_i = sech2_phi / (2.0 * rad_safe * epsilon)

        # BUG FIX 2: sign correction.
        #
        # Differentiating u_i = 0.5*(1 - tanh(phi)) analytically in polar
        # coordinates gives:
        #
        #   ∇²u_i = d²u/dr² + (1/r)*du/dr + (1/r²)*d²u/dθ²
        #
        #   d²u/dr²      =  tanh*sech²/ε²                      = +s3
        #   (1/r)*du/dr  = -sech²/(2*r*ε)                      = -s4
        #   d²u/dθ²/r²   = (p²*tanh*sech²/ε² + s2) / r²       = (s1+s2)/r²
        #
        #   => ∇²u_i = (s1 + s2)/r² + s3 - s4
        #
        # The original code had -(s1+s2)/r² - s3 + s4, which is the negative
        # of the correct expression.  The paper's eq. 4.4 writes (-s1+s2)/r²
        # which appears to use the negative-Laplacian convention (Δu = -f),
        # but the first Poisson problem in the same paper (eq. 4.1) shows that
        # ∇²u = f directly, so the formula below is self-consistent with the
        # exact solution defined here.
        F += (s1_i + s2_i) / (rad_safe ** 2) + s3_i - s4_i

    return F


PROBLEM = PoissonTestProblem(
    name="polar_star",
    domain={
        "xa": -1.0,
        "xb": 1.0,
        "ya": -1.0,
        "yb": 1.0,
    },
    constants={
        "epsilon": 0.01,
    },
    rhs=rhs,
    exact=exact,
)