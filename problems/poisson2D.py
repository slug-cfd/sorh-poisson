import numpy as np
import math

# import BC_2D as bc
from methods import poisson as pm

# from typing import Callable, Optional


# left "blob"
def A(x, y, R):
    c = 3.0 * (R ** 2)
    return (((x - (5.0 * R)) ** 2) + (y ** 2)) / c


# right "blob"
def B(x, y, R):
    c = 3.0 * (R ** 2)
    return (((x + (5.0 * R)) ** 2) + (y ** 2)) / c


# 2D density distribution based on the form of phi(x, y).
# the paper defined phi(x, y, z), so we simply drop the z terms.
def rho(x, y, rho0, R):
    c = rho0 / 3.0
    a = A(x, y, R)
    b = B(x, y, R)

    left = (2 - a) * ((1 + a) ** (-5.0 / 2.0))
    right = (2 - b) * ((1 + b) ** (-5.0 / 2.0))

    return c * (left + right)


# phi(x, y, z) from paper -> drop z terms -> phi(x, y)
def phi_exact(x, y, rho0, R, G):
    c = -4.0 * math.pi * G * rho0 * (R ** 2)

    left = (1 + A(x, y, R)) ** (-0.5)
    right = (1 + B(x, y, R)) ** (-0.5)

    return c * (left + right)


# here we assume 4*pi*G = 1 (as the paper did),
# thus ∆phi = f(x, y) where f(x, y) = rho(x, y)
def f(x, y, rho0, R):
    return rho(x, y, rho0, R)


def get_all_f_from_xy(params) -> np.ndarray:
    x, y, rho0, R = params.x, params.y, params.rho0, params.R

    X, Y = np.meshgrid(x, y, indexing='ij')  # shape (Nx+2*ngc, Ny+2*ngc)
    return f(X, Y, rho0, R)


def solve(params):
    match params.method:
        case "parabolic":
            U, t, iter_count, res, err_conv = pm.parabolic(params, phi_exact, get_all_f_from_xy)
        case "DS1":
            U, t, iter_count, res, err_conv = pm.DS1(params, phi_exact, get_all_f_from_xy)
        case "DS2":
            U, t, iter_count, res, err_conv = pm.DS2(params, phi_exact, get_all_f_from_xy)
        case "unsplit":
            U, t, iter_count, res, err_conv = pm.unsplit(params, phi_exact, get_all_f_from_xy)
        case _:
            return -1

    return U, t, iter_count, res, err_conv
