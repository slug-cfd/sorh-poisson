# multigrid.py

from __future__ import annotations

import time

from methods.Poisson.poisson_helpers import *
import BC_2D as bc
from config import Config

# ============================================================
# Basic utilities
# ============================================================
def coarse_full_size(N_f: int) -> int:
    n_f = N_f - 2          # interior count
    n_c = (n_f - 1) // 2   # standard nested coarsening
    return n_c + 2         # back to full size


def zero_like_interior(U: np.ndarray) -> np.ndarray:
    """Return a zero array with the same shape/type as U."""
    return np.zeros_like(U)


def copy_boundaries_from(U_src: np.ndarray, U_dst: np.ndarray, boundaries, config):
    """
    Optional helper if your environment needs explicit ghost/boundary updates.

    In many codes, boundary values/ghost cells are applied by a separate routine.
    If so, replace this with your own logic or remove it.
    """
    U_dst[...] = U_src
    return U_dst


# ============================================================
# Residual
# ============================================================

def apply_operator(U: np.ndarray, boundaries, config) -> np.ndarray:
    """
    Return L(U) on the full grid, including interior locations.

    IMPORTANT:
    Replace this with your actual discrete Poisson operator evaluation.
    The convention assumed here is that your PDE is

        L(U) = f

    for example L(U) = U_xx + U_yy.
    """
    return get_all_laplace(U, config)   # centered 2D laplacian


def compute_residual(U: np.ndarray, boundaries, config) -> np.ndarray:
    """
    Compute residual r = f - L(U).

    Assumes config.f_all is defined on the same full grid as U, including ghost
    cells if that is how your environment stores arrays.
    """
    LU = apply_operator(U, boundaries, config)
    r = LU - config.f_all
    return r


def residual_norm(U: np.ndarray, boundaries, config):
    """Convenience function for monitoring convergence."""
    norm, ib, ie, jb, je = config.norm, config.ib, config.ie, config.jb, config.je
    Nx, Ny = config.Nx, config.Ny

    r = compute_residual(U, boundaries, config)
    # r_int = r[ib:ie + 1, jb:je + 1]

    # R = L - f_all
    res_residual = np.linalg.norm(r[ib:ie + 1, jb:je + 1], ord=norm)
    if norm in [1, 2, "fro"]:
        res_residual /= np.sqrt(Nx * Ny)

    # if norm == np.inf:
    #     return np.max(np.abs(r_int))
    # return np.linalg.norm(r_int, ord=norm)
    return res_residual


# ============================================================
# Restriction / prolongation
# ============================================================

def restrict_injection(F: np.ndarray) -> np.ndarray:
    """
    Simple injection restriction for full grids with one ghost cell.

    Assumes odd-numbered interior counts so that coarsening by factor ~2 makes sense.

    Fine interior indices: 1...N
    Coarse interior samples every other fine point.
    """
    Nc_x = (F.shape[0] - 2 + 1) // 2
    Nc_y = (F.shape[1] - 2 + 1) // 2

    Fc = np.zeros((Nc_x + 2, Nc_y + 2), dtype=F.dtype)

    Fc[1:-1, 1:-1] = F[1:-1:2, 1:-1:2]  # copy every other point from fine grid
    return Fc


def restrict_full_weighting(F: np.ndarray) -> np.ndarray:
    """
    Standard 2D full-weighting restriction for cell-centered or pointwise interior grids
    stored with one ghost layer.

    Coarse interior point corresponds to every other fine interior point.
    """
    Nfx = F.shape[0] - 2
    Nfy = F.shape[1] - 2

    Ncx = (Nfx + 1) // 2
    Ncy = (Nfy + 1) // 2

    Fc = np.zeros((Ncx + 2, Ncy + 2), dtype=F.dtype)

    for I in range(1, Ncx + 1):
        for J in range(1, Ncy + 1):
            i = 2 * I - 1
            j = 2 * J - 1

            Fc[I, J] = (
                4.0 * F[i, j]
                + 2.0 * (F[i - 1, j] + F[i + 1, j] + F[i, j - 1] + F[i, j + 1])
                + (F[i - 1, j - 1] + F[i - 1, j + 1] + F[i + 1, j - 1] + F[i + 1, j + 1])
            ) / 16.0

    return Fc


def prolong_bilinear(Ec: np.ndarray, fine_shape: tuple[int, int]) -> np.ndarray:
    """
    Bilinear prolongation from coarse full grid Ec to fine full grid Ef.

    Supports fine interior sizes satisfying either
        n_f = 2*n_c - 1
    or
        n_f = 2*n_c
    in each direction,
    which is consistent with the restriction rule
        n_c = (n_f + 1) // 2

    Both Ec and Ef include one ghost-cell layer on each side.
    """
    Ef = np.zeros(fine_shape, dtype=Ec.dtype)

    nxc = Ec.shape[0] - 2
    nyc = Ec.shape[1] - 2
    nxf = fine_shape[0] - 2
    nyf = fine_shape[1] - 2

    valid_x = (nxf == 2 * nxc - 1) or (nxf == 2 * nxc)
    valid_y = (nyf == 2 * nyc - 1) or (nyf == 2 * nyc)

    if not (valid_x and valid_y):
        raise ValueError(
            "Incompatible grid sizes for prolongation: "
            f"coarse interior = ({nxc}, {nyc}), "
            f"fine interior = ({nxf}, {nyf}), "
            f"expected each fine interior dimension to be either "
            f"2*nc-1 or 2*nc."
        )

    # --------------------------------------------------
    # 1) Inject coarse points
    # coarse (I,J) -> fine (2I-1, 2J-1)
    # --------------------------------------------------
    for I in range(1, nxc + 1):
        for J in range(1, nyc + 1):
            i = 2 * I - 1
            j = 2 * J - 1
            if i <= nxf and j <= nyf:
                Ef[i, j] = Ec[I, J]

    # --------------------------------------------------
    # 2) Interpolate in x-direction
    # even i, odd j
    # interior between coarse points
    # --------------------------------------------------
    for I in range(1, nxc):
        for J in range(1, nyc + 1):
            i = 2 * I
            j = 2 * J - 1
            if i <= nxf and j <= nyf:
                Ef[i, j] = 0.5 * (Ec[I, J] + Ec[I + 1, J])

    # If fine grid has an extra last row (nxf = 2*nxc),
    # fill it by copying the last coarse row.
    if nxf == 2 * nxc:
        i = nxf
        for J in range(1, nyc + 1):
            j = 2 * J - 1
            if j <= nyf:
                Ef[i, j] = Ec[nxc, J]

    # --------------------------------------------------
    # 3) Interpolate in y-direction
    # odd i, even j
    # --------------------------------------------------
    for I in range(1, nxc + 1):
        for J in range(1, nyc):
            i = 2 * I - 1
            j = 2 * J
            if i <= nxf and j <= nyf:
                Ef[i, j] = 0.5 * (Ec[I, J] + Ec[I, J + 1])

    # If fine grid has an extra last column (nyf = 2*nyc),
    # fill it by copying the last coarse column.
    if nyf == 2 * nyc:
        j = nyf
        for I in range(1, nxc + 1):
            i = 2 * I - 1
            if i <= nxf:
                Ef[i, j] = Ec[I, nyc]

    # --------------------------------------------------
    # 4) Interpolate centers
    # even i, even j
    # --------------------------------------------------
    for I in range(1, nxc):
        for J in range(1, nyc):
            i = 2 * I
            j = 2 * J
            if i <= nxf and j <= nyf:
                Ef[i, j] = 0.25 * (
                    Ec[I, J]
                    + Ec[I + 1, J]
                    + Ec[I, J + 1]
                    + Ec[I + 1, J + 1]
                )

    # Extra last row of center-type points if nxf = 2*nxc
    if nxf == 2 * nxc:
        i = nxf
        for J in range(1, nyc):
            j = 2 * J
            if j <= nyf:
                Ef[i, j] = 0.5 * (Ec[nxc, J] + Ec[nxc, J + 1])

    # Extra last column of center-type points if nyf = 2*nyc
    if nyf == 2 * nyc:
        j = nyf
        for I in range(1, nxc):
            i = 2 * I
            if i <= nxf:
                Ef[i, j] = 0.5 * (Ec[I, nyc] + Ec[I + 1, nyc])

    # Bottom-right extra corner if both dimensions are even
    if nxf == 2 * nxc and nyf == 2 * nyc:
        Ef[nxf, nyf] = Ec[nxc, nyc]

    return Ef


# ============================================================
# Coarse-grid config construction
# ============================================================
def coarsen_boundaries(fine_boundaries, coarse_config, mode="error"):
    """
    Build coarse-grid boundary arrays consistent with BC_2D.py.

    Parameters
    ----------
    fine_boundaries : ignored
        Included only for interface compatibility; with your current setup
        it is more consistent to rebuild boundaries from coarse_config.
    coarse_config : Config
        Coarse-grid config.
    mode : {"error", "solution"}
        "error"    -> homogeneous Dirichlet BCs for coarse-grid correction
        "solution" -> physical Dirichlet BCs rebuilt from phi_exact on coarse grid

    Returns
    -------
    boundaries : list[np.ndarray]
        [phi_xL, phi_xR, phi_yB, phi_yT]
    """
    if mode == "error":
        # For the error equation, use homogeneous Dirichlet BCs.
        phi_xL = np.zeros_like(coarse_config.y)
        phi_xR = np.zeros_like(coarse_config.y)
        phi_yB = np.zeros_like(coarse_config.x)
        phi_yT = np.zeros_like(coarse_config.x)
        return [phi_xL, phi_xR, phi_yB, phi_yT]

    elif mode == "solution":
        # For FMG / original equation solves, rebuild the physical BCs
        # exactly the same way your top-level code does it now.
        return bc.get_bcs_poisson_2d(coarse_config, coarse_config.phi_exact)

    else:
        raise ValueError(f"Unknown boundary mode: {mode}")


def build_coarse_config(fine_config, coarse_f_all, nx_coarse=None, ny_coarse=None):
    """
    Build a coarse Config object consistent with the current Config implementation.

    Parameters
    ----------
    fine_config : Config
        Fine-grid config.
    coarse_f_all : np.ndarray
        Coarse-grid RHS array to store in coarse_config.f_all.
        This should already have the full coarse-grid shape, matching coarse U.
    nx_coarse, ny_coarse : int or None
        Total coarse-grid sizes INCLUDING ghost layers.
        If omitted, infer from coarse_f_all.shape.

    Returns
    -------
    coarse_config : Config
    """
    if nx_coarse is None or ny_coarse is None:
        nx_coarse, ny_coarse = coarse_f_all.shape

    coarse_config = Config(
        problem=fine_config.problem,
        poisson_problem=fine_config.poisson_problem,
        method=fine_config.method,
        bc_type=fine_config.bc_type,
        Nx=nx_coarse,
        Ny=ny_coarse,
        cfl=fine_config.cfl,
        omega=fine_config.omega,
        iter_max=fine_config.iter_max,
        norm=fine_config.norm,
        f=fine_config.f,
        phi_exact=fine_config.phi_exact,
        mg_config=fine_config.mg_config,
        h_unsplit_config=fine_config.h_unsplit_config,
        multigrid=fine_config.multigrid,
    )

    # Override the automatically built RHS with the one actually needed
    # for the coarse-grid problem.
    coarse_config.f_all = coarse_f_all.copy()

    # Optional but often useful:
    # if you ever want the exact solution sampled on the coarse grid, __post_init__
    # already builds it from phi_exact, so nothing else is needed here.

    return coarse_config

# ============================================================
# Smoothing
# ============================================================

def smooth(U: np.ndarray, params, boundaries, config, full_sweep, num_sweeps: int) -> np.ndarray:
    """
    Apply the user-provided iterative method several times.

    full_sweep is expected to perform one full-grid iteration:
        U_new = full_sweep(U, params, boundaries, config)

    If your full_sweep mutates U in-place instead, adapt accordingly.
    """
    for _ in range(num_sweeps):
        U = full_sweep(U, params, boundaries, config)
    return U


# ============================================================
# Coarsest-level solve
# ============================================================

def coarse_relaxation_solve(
    params,
    boundaries,
    config,
    full_sweep,
):
    """
    Approximate coarse solve by repeated smoothing.

    This is the easiest initial implementation and is often sufficient for a first
    multigrid version.
    """
    mg = config.mg_config
    max_sweeps, tol = mg.coarse_max_sweeps, mg.coarse_tol

    U = config.U0.copy()

    for _ in range(max_sweeps):
        U = full_sweep(U, params, boundaries, config)
        rn = residual_norm(U, boundaries, config)
        if rn < tol:
            break

    return U


# ============================================================
# V-cycle
# ============================================================

def v_cycle(
    U: np.ndarray,
    params,
    boundaries,
    config,
    full_sweep,
):
    """
    Perform one multigrid V-cycle on the equation L(U)=f.

    Parameters
    ----------
    U : current approximation on this level
    params : solver params for full_sweep
    boundaries : boundary object for this level
    config : config for this level
    full_sweep : one-step smoother
    nu1, nu2 : pre- / post-smoothing sweeps
    min_coarse_n : stop recursion when min(nx, ny) <= this
    """

    mg = config.mg_config
    min_coarse_n, restriction, prolongation = mg.min_coarse_n, mg.restriction, mg.prolongation
    nu1, nu2 = mg.nu1, mg.nu2

    nx = U.shape[0] - 2
    ny = U.shape[1] - 2

    # Base case: solve approximately on coarsest level
    if min(nx, ny) <= min_coarse_n:
        return coarse_relaxation_solve(
            params=params,
            boundaries=boundaries,
            config=config,
            full_sweep=full_sweep,
        )

    # --------------------------------------------------------
    # 1. Pre-smoothing on fine grid
    # --------------------------------------------------------
    U = smooth(U, params, boundaries, config, full_sweep, num_sweeps=nu1)

    # --------------------------------------------------------
    # 2. Residual on fine grid
    # --------------------------------------------------------
    r_f = compute_residual(U, boundaries, config)

    # --------------------------------------------------------
    # 3. Restrict residual to coarse grid
    # --------------------------------------------------------
    if restriction == "injection":
        r_c = restrict_injection(r_f)
    elif restriction == "full_weighting":
        r_c = restrict_full_weighting(r_f)
    else:
        raise ValueError(f"Unknown restriction: {restriction}")

    # --------------------------------------------------------
    # 4. Recursive coarse-grid error solve:
    #       L_c(e_c) = -r_c
    # --------------------------------------------------------
    coarse_f_all = -r_c

    coarse_config = build_coarse_config(config, coarse_f_all=coarse_f_all)
    coarse_boundaries = coarsen_boundaries(boundaries, coarse_config, mode="error")

    e_c0 = coarse_config.U0.copy()
    bc.apply_bcs_poisson_2d(e_c0, coarse_config, coarse_boundaries)

    e_c = v_cycle(
        U=e_c0,
        params=params,
        boundaries=coarse_boundaries,
        config=coarse_config,
        full_sweep=full_sweep,
    )

    # --------------------------------------------------------
    # 5. Prolong and correct
    # --------------------------------------------------------
    if prolongation == "bilinear":
        e_f = prolong_bilinear(e_c, U.shape)
    else:
        raise ValueError(f"Unknown prolongation: {prolongation}")

    U = U + e_f

    bc.apply_bcs_poisson_2d(U, config, boundaries)

    # --------------------------------------------------------
    # 6. Post-smoothing
    # --------------------------------------------------------
    U = smooth(U, params, boundaries, config, full_sweep, num_sweeps=nu2)

    return U


# ============================================================
# Repeated V-cycles until convergence
# ============================================================
def solve_multigrid(
    params,
    boundaries,
    config,
    full_sweep,
    verbose: bool = True,
):
    """
    Solve using repeated V-cycles.

    Stopping criterion:
        primary = update norm ||U - U_old||

    Histories returned in a format similar to the standard iterative solver.
    """

    # ------------------------------------------------------------
    # 1. setup
    # ------------------------------------------------------------
    ib, ie, jb, je = config.ib, config.ie, config.jb, config.je
    Nx, Ny = config.Nx, config.Ny
    norm = config.norm
    phi_exact = config.phi_exact
    f_all = config.f_all
    U_exact = config.U_exact

    mg = config.mg_config
    max_cycles, tol_update, res_max = mg.max_cycles, mg.tol_update, mg.res_max

    # initialize solution
    U = config.U0.copy()
    bc.apply_bcs_poisson_2d(U, config, boundaries)

    # histories
    update_hist = []       # ||U - U_old||
    residual_hist = []     # ||L(U) - f||
    exact_hist = []        # ||U - U_exact||
    cycle_time_hist = []   # elapsed time per V-cycle

    # best iterate tracking (diagnostic only)
    best_exact = np.inf
    U_best = U.copy()
    best_iter = 0

    iter_count = 0
    t = 0.0

    t0 = time.perf_counter()

    # ------------------------------------------------------------
    # 2. V-cycle loop
    # ------------------------------------------------------------
    while iter_count < max_cycles:
        tc = time.perf_counter()

        U_old = U.copy()

        U = v_cycle(
            U=U,
            params=params,
            boundaries=boundaries,
            config=config,
            full_sweep=full_sweep,
        )

        # Re-apply BCs on the corrected fine-grid solution
        bc.apply_bcs_poisson_2d(U, config, boundaries)

        cycle_time = time.perf_counter() - tc
        cycle_time_hist.append(cycle_time)

        # -------------------------
        # update norm (primary stopping criterion)
        # -------------------------
        dU = U - U_old
        res_update = np.linalg.norm(dU[ib:ie+1, jb:je+1], ord=norm)
        if norm in [1, 2, "fro"]:
            res_update /= np.sqrt(Nx * Ny)

        update_hist.append(res_update)

        # -------------------------
        # residual norm (diagnostic)
        # -------------------------
        if f_all is not None:
            L = get_all_laplace(U, config)
            R = L - f_all
            res_residual = np.linalg.norm(R[ib:ie+1, jb:je+1], ord=norm)
            if norm in [1, 2, "fro"]:
                res_residual /= np.sqrt(Nx * Ny)
        else:
            res_residual = np.nan

        residual_hist.append(res_residual)

        # -------------------------
        # exact error (diagnostic)
        # -------------------------
        if phi_exact is not None:
            err = U - U_exact
            res_exact = np.linalg.norm(err[ib:ie+1, jb:je+1], ord=norm)
            if norm in [1, 2, "fro"]:
                res_exact /= np.sqrt(Nx * Ny)

            exact_hist.append(res_exact)

            if res_exact < best_exact:
                best_exact = res_exact
                U_best = U.copy()
                best_iter = iter_count
        else:
            exact_hist.append(np.nan)

        if verbose:
            print(
                f"[MG] cycle = {iter_count+1:3d} | "
                f"update = {res_update:.6e} | "
                f"residual = {res_residual:.6e}"
            )

        # -------------------------
        # stopping conditions
        # -------------------------
        converged_update = res_update < tol_update

        # primary requested stopping criterion:
        if converged_update:
            break

        if not np.isfinite(res_update) or res_update > res_max:
            break

        t += cycle_time
        iter_count += 1

    elapsed = time.perf_counter() - t0

    # if loop exits immediately after break, iter_count still counts completed
    # cycles starting from 0, so fix to actual number of stored entries
    iter_count = len(update_hist)

    # ------------------------------------------------------------
    # 3. prepare return
    # ------------------------------------------------------------
    histories = {
        "update": np.array(update_hist),
        "residual": np.array(residual_hist),
        "exact": np.array(exact_hist),
        "cycle_time": np.array(cycle_time_hist),
        "best_exact": best_exact,
        "best_iter": best_iter,
        "elapsed": elapsed,
        "cycles": iter_count,
        "U_best": U_best,
    }

    # ------------------------------------------------------------
    # 4. return
    # ------------------------------------------------------------
    return U, t, iter_count, histories


# ============================================================
# Full multigrid (FMG)
# ============================================================

def fmg_cycle(
    params,
    boundaries,
    config,
    full_sweep,
):
    """
    Full Multigrid (FMG) recursive initialization.

    Strategy:
      1. If this is the coarsest grid, solve approximately on this level.
      2. Otherwise, restrict the ORIGINAL RHS to a coarser grid.
      3. Recursively solve the coarse problem for the solution U_c.
      4. Prolong U_c to this level to obtain an initial guess.
      5. Apply one V-cycle on this level to refine.

    Returns
    -------
    U : np.ndarray
        Approximate solution on the current grid.
    """

    mg = config.mg_config
    min_coarse_n, restriction, prolongation = mg.min_coarse_n, mg.restriction, mg.prolongation
    coarse_max_sweeps, coarse_tol = mg.coarse_max_sweeps, mg.coarse_tol

    nx_int = config.Nx - 2
    ny_int = config.Ny - 2

    # ------------------------------------------------------------
    # Base case: approximately solve directly on the coarsest grid
    # ------------------------------------------------------------
    if min(nx_int, ny_int) <= min_coarse_n:
        U = config.U0.copy()
        bc.apply_bcs_poisson_2d(U, config, boundaries)

        for _ in range(coarse_max_sweeps):
            U_old = U.copy()
            U = full_sweep(U, params, boundaries, config)
            bc.apply_bcs_poisson_2d(U, config, boundaries)

            dU = U - U_old
            res_update = np.linalg.norm(
                dU[config.ib:config.ie+1, config.jb:config.je+1],
                ord=config.norm
            )
            if config.norm in [1, 2, "fro"]:
                res_update /= np.sqrt(config.Nx * config.Ny)

            if res_update < coarse_tol:
                break

        return U

    # ------------------------------------------------------------
    # Restrict the original RHS to the next coarser grid
    # ------------------------------------------------------------
    if restriction == "injection":
        f_c = restrict_injection(config.f_all)
    elif restriction == "full_weighting":
        f_c = restrict_full_weighting(config.f_all)
    else:
        raise ValueError(f"Unknown restriction: {restriction}")

    coarse_config = build_coarse_config(config, coarse_f_all=f_c)
    coarse_boundaries = coarsen_boundaries(boundaries, coarse_config, mode="solution")

    # ------------------------------------------------------------
    # Recursively solve the coarse-grid solution problem
    # ------------------------------------------------------------
    U_c = fmg_cycle(
        params=params,
        boundaries=coarse_boundaries,
        config=coarse_config,
        full_sweep=full_sweep,
    )

    # ------------------------------------------------------------
    # Prolong coarse solution to this level
    # ------------------------------------------------------------
    if prolongation == "bilinear":
        U = prolong_bilinear(U_c, config.U0.shape)
    else:
        raise ValueError(f"Unknown prolongation: {prolongation}")

    bc.apply_bcs_poisson_2d(U, config, boundaries)

    # ------------------------------------------------------------
    # Refine with one V-cycle on this level
    # ------------------------------------------------------------
    U = v_cycle(
        U=U,
        params=params,
        boundaries=boundaries,
        config=config,
        full_sweep=full_sweep,
    )

    bc.apply_bcs_poisson_2d(U, config, boundaries)

    return U


# ============================================================
# FMG solve wrapper with histories / stopping like solve_multigrid
# ============================================================
def solve_fmg(
    params,
    boundaries,
    config,
    full_sweep,
    verbose: bool = True,
):
    """
    Solve using FMG initialization followed by repeated V-cycles until convergence.

    This mirrors the updated solve_multigrid() structure:
      - same update-based stopping criterion
      - same histories layout
      - same return signature

    Returns
    -------
    U, t, iter_count, histories
    """

    # ------------------------------------------------------------
    # 1. setup
    # ------------------------------------------------------------
    ib, ie, jb, je = config.ib, config.ie, config.jb, config.je
    Nx, Ny = config.Nx, config.Ny
    norm = config.norm
    phi_exact = config.phi_exact
    f_all = config.f_all
    U_exact = config.U_exact

    mg = config.mg_config
    max_cycles, tol_update, res_max = mg.max_cycles, mg.tol_update, mg.res_max

    # histories
    update_hist = []
    residual_hist = []
    exact_hist = []
    cycle_time_hist = []

    # best iterate tracking
    best_exact = np.inf
    U_best = None
    best_iter = 0

    iter_count = 0
    t = 0.0

    t0 = time.perf_counter()

    # ------------------------------------------------------------
    # 2. FMG initialization
    # ------------------------------------------------------------
    tfmg0 = time.perf_counter()

    U = fmg_cycle(
        params=params,
        boundaries=boundaries,
        config=config,
        full_sweep=full_sweep,
    )

    bc.apply_bcs_poisson_2d(U, config, boundaries)

    tfmg = time.perf_counter() - tfmg0

    # ------------------------------------------------------------
    # 3. Record FMG initialization as cycle 1-equivalent entry
    # ------------------------------------------------------------
    # Use U0 as reference for the first update metric
    U0_ref = config.U0.copy()
    bc.apply_bcs_poisson_2d(U0_ref, config, boundaries)

    dU = U - U0_ref
    res_update = np.linalg.norm(dU[ib:ie+1, jb:je+1], ord=norm)
    if norm in [1, 2, "fro"]:
        res_update /= np.sqrt(Nx * Ny)
    update_hist.append(res_update)

    if f_all is not None:
        L = get_all_laplace(U, config)
        R = L - f_all
        res_residual = np.linalg.norm(R[ib:ie+1, jb:je+1], ord=norm)
        if norm in [1, 2, "fro"]:
            res_residual /= np.sqrt(Nx * Ny)
    else:
        res_residual = np.nan
    residual_hist.append(res_residual)

    if phi_exact is not None:
        err = U - U_exact
        res_exact = np.linalg.norm(err[ib:ie+1, jb:je+1], ord=norm)
        if norm in [1, 2, "fro"]:
            res_exact /= np.sqrt(Nx * Ny)
        exact_hist.append(res_exact)

        best_exact = res_exact
        U_best = U.copy()
        best_iter = 0
    else:
        exact_hist.append(np.nan)
        U_best = U.copy()

    cycle_time_hist.append(tfmg)

    if verbose:
        print(
            f"[FMG] cycle = {1:3d} | "
            f"update = {res_update:.6e} | "
            f"residual = {res_residual:.6e}"
        )

    t += tfmg
    iter_count = 1

    # ------------------------------------------------------------
    # 4. Repeated V-cycles after FMG initialization
    # ------------------------------------------------------------
    while iter_count < max_cycles:
        tc = time.perf_counter()

        U_old = U.copy()

        U = v_cycle(
            U=U,
            params=params,
            boundaries=boundaries,
            config=config,
            full_sweep=full_sweep,
        )

        bc.apply_bcs_poisson_2d(U, config, boundaries)

        cycle_time = time.perf_counter() - tc
        cycle_time_hist.append(cycle_time)

        # -------------------------
        # update norm (primary stopping criterion)
        # -------------------------
        dU = U - U_old
        res_update = np.linalg.norm(dU[ib:ie+1, jb:je+1], ord=norm)
        if norm in [1, 2, "fro"]:
            res_update /= np.sqrt(Nx * Ny)
        update_hist.append(res_update)

        # -------------------------
        # residual norm (diagnostic)
        # -------------------------
        if f_all is not None:
            L = get_all_laplace(U, config)
            R = L - f_all
            res_residual = np.linalg.norm(R[ib:ie+1, jb:je+1], ord=norm)
            if norm in [1, 2, "fro"]:
                res_residual /= np.sqrt(Nx * Ny)
        else:
            res_residual = np.nan
        residual_hist.append(res_residual)

        # -------------------------
        # exact error (diagnostic)
        # -------------------------
        if phi_exact is not None:
            err = U - U_exact
            res_exact = np.linalg.norm(err[ib:ie+1, jb:je+1], ord=norm)
            if norm in [1, 2, "fro"]:
                res_exact /= np.sqrt(Nx * Ny)
            exact_hist.append(res_exact)

            if res_exact < best_exact:
                best_exact = res_exact
                U_best = U.copy()
                best_iter = iter_count
        else:
            exact_hist.append(np.nan)

        if verbose:
            print(
                f"[FMG] cycle = {iter_count+1:3d} | "
                f"update = {res_update:.6e} | "
                f"residual = {res_residual:.6e}"
            )

        # -------------------------
        # stopping conditions
        # -------------------------
        converged_update = res_update < tol_update

        if converged_update:
            break

        if not np.isfinite(res_update) or res_update > res_max:
            break

        t += cycle_time
        iter_count += 1

    elapsed = time.perf_counter() - t0
    iter_count = len(update_hist)

    # ------------------------------------------------------------
    # 5. histories / return
    # ------------------------------------------------------------
    histories = {
        "update": np.array(update_hist),
        "residual": np.array(residual_hist),
        "exact": np.array(exact_hist),
        "cycle_time": np.array(cycle_time_hist),
        "best_exact": best_exact,
        "best_iter": best_iter,
        "elapsed": elapsed,
        "cycles": iter_count,
        "U_best": U_best,
    }

    return U, t, iter_count, histories

def solve(config, get_params, full_sweep):
    mg = config.mg_config
    ### mg_config structure ###
    # @dataclass
    # class MultiGridConfig:
    #     solver: str = "fmg"  # "rep_v_cycles" or "fmg"
    #     max_cycles: int = 200,
    #     tol_update: float = 1e-5,
    #     tol_residual: float = 1e-5,
    #     res_max: float = 1e8,
    #     nu1: int = 2,
    #     nu2: int = 2,
    #     min_coarse_n: int = 4,
    #     coarse_max_sweeps: int = 100,
    #     coarse_tol: float = 1e-6,
    #     restriction: str = "full_weighting",
    #     prolongation: str = "bilinear",

    solver = mg.solver
    params = get_params(config)

    # initialization
    boundaries = bc.get_bcs_poisson_2d(config, config.phi_exact)

    U = config.U0.copy()
    bc.apply_bcs_poisson_2d(U, config, boundaries)

    ### N = 64 parameters ###

    # p_unsplit
    # nu1, nu2 = 2, 2
    # coarse_max_sweeps = 200
    # min_coarse_n = 2
    # coarse_tol = 1e-6

    # h_unsplit
    # nu1, nu2 = 4, 4
    # coarse_max_sweeps = 400
    # min_coarse_n = 4
    # coarse_tol = 1e-6

    if solver == "rep_v_cycles":
        U, t, cycle_count, histories = solve_multigrid(params, boundaries, config, full_sweep,)
    elif solver == "fmg":
        U, t, cycle_count, histories = solve_fmg(params, boundaries, config, full_sweep,)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    return U, t, cycle_count, histories
