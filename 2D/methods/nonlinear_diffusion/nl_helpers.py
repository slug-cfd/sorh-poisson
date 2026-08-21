"""Shared nonlinear-diffusion face and sweep helpers."""

import numpy as np


def _face_velocity_scales(vx, vy):
    vx_max = float(np.max(np.abs(vx)))
    vy_max = float(np.max(np.abs(vy)))
    if not np.isfinite(vx_max) or not np.isfinite(vy_max):
        raise FloatingPointError("Nonlinear face velocity is nonfinite.")
    return vx_max, vy_max


def adaptive_parabolic_dt(vx, vy, config):
    vx_max, vy_max = _face_velocity_scales(vx, vy)
    v_max = max(vx_max, vy_max)
    if v_max <= 1e-14:
        raise FloatingPointError("Cannot form nonlinear parabolic CFL step: all face velocities are zero.")
    return config.cfl / (2.0 * v_max * (1.0 / config.dx**2 + 1.0 / config.dy**2))


def adaptive_hyperbolic_dt(vx, vy, config):
    vx_max, vy_max = _face_velocity_scales(vx, vy)
    candidates = []
    if vx_max > 1e-14:
        candidates.append(config.dx / vx_max)
    if vy_max > 1e-14:
        candidates.append(config.dy / vy_max)
    if not candidates:
        raise FloatingPointError("Cannot form nonlinear hyperbolic CFL step: all face velocities are zero.")
    return config.cfl * min(candidates)


def get_vx(U, params):
    Nx, Ny = params.Nx, params.Ny
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
    coefficient = params.problem_instance.coefficient

    vx = np.zeros((Nx, Ny, 2), dtype=np.float64)

    for j in range(jb, je + 1):
        for i in range(ib, ie + 1):
            # update values for v
            um = 0.5 * (U[i - 1, j] + U[i, j])
            up = 0.5 * (U[i, j] + U[i + 1, j])
            vm = coefficient(um, params)
            vp = coefficient(up, params)
            vx[i - ib, j - jb, 0] = vm
            vx[i - ib, j - jb, 1] = vp

    return vx


def get_vx_point(U, i, j, config):
    um = 0.5 * (U[i - 1, j] + U[i, j])
    up = 0.5 * (U[i, j] + U[i + 1, j])
    coefficient = config.problem_instance.coefficient
    vm = coefficient(um, config)
    vp = coefficient(up, config)
    return vm, vp


def get_vy(U, params):
    Nx, Ny = params.Nx, params.Ny
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je
    coefficient = params.problem_instance.coefficient

    vy = np.zeros((Nx, Ny, 2), dtype=np.float64)

    for i in range(ib, ie + 1):
        for j in range(jb, je + 1):
            # update values for v
            um = 0.5 * (U[i, j - 1] + U[i, j])
            up = 0.5 * (U[i, j] + U[i, j + 1])
            vm = coefficient(um, params)
            vp = coefficient(up, params)
            vy[i - ib, j - jb, 0] = vm
            vy[i - ib, j - jb, 1] = vp

    return vy


def get_vy_point(U, i, j, config):
    um = 0.5 * (U[i, j - 1] + U[i, j])
    up = 0.5 * (U[i, j] + U[i, j + 1])
    coefficient = config.problem_instance.coefficient
    vm = coefficient(um, config)
    vp = coefficient(up, config)
    return vm, vp


def nonlinear_flux_residual(U, config):
    """Return the interior residual of div(v(U) grad(U)) = f."""
    ib, ie, jb, je = config.ib, config.ie, config.jb, config.je
    center = U[ib:ie + 1, jb:je + 1]
    left = U[ib - 1:ie, jb:je + 1]
    right = U[ib + 1:ie + 2, jb:je + 1]
    bottom = U[ib:ie + 1, jb - 1:je]
    top = U[ib:ie + 1, jb + 1:je + 2]

    coefficient = config.problem_instance.coefficient
    v_left = coefficient(0.5 * (left + center), config)
    v_right = coefficient(0.5 * (center + right), config)
    v_bottom = coefficient(0.5 * (bottom + center), config)
    v_top = coefficient(0.5 * (center + top), config)

    residual = (
        (v_right * (right - center) - v_left * (center - left))
        / config.dx**2
        + (v_top * (top - center) - v_bottom * (center - bottom))
        / config.dy**2
    )
    if config.f_all is not None:
        residual -= config.f_all[ib:ie + 1, jb:je + 1]
    return residual
