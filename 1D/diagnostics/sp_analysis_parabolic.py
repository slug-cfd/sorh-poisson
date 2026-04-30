import numpy as np
import matplotlib.pyplot as plt


def parabolic_gs_sor_matrix(N, r, omega):
    """
    Spectral-radius iteration matrix for 1D parabolic GS-type SOR.

    Grid:
        N intervals, N+1 total points: x_0, ..., x_N
        Interior unknowns: x_1, ..., x_{N-1}

    PDE:
        u_t = v u_xx

    GS-type relaxed update:
        u_i^{n+1}
        =
        u_i^n + omega*r*(u_{i-1}^{n+1} - 2*u_i^n + u_{i+1}^n)

    where:
        r = v*dt/dx^2

    Matrix form:
        L u^{n+1} = omega R u^n + h

    with:
        L = tridiag(-omega*r, 1, 0)
        R = tridiag(0, 1/omega - 2r, r)

    The iteration matrix is:
        G = L^{-1} omega R
    """
    m = N - 1

    L = np.eye(m)
    for i in range(1, m):
        L[i, i - 1] = -omega * r

    R = np.zeros((m, m))
    for i in range(m):
        R[i, i] = 1.0 / omega - 2.0 * r

        if i < m - 1:
            R[i, i + 1] = r

    G = np.linalg.solve(L, omega * R)

    return G


def spectral_radius(N, r, omega):
    """
    Compute rho(G) for the 1D parabolic GS-type SOR iteration matrix.
    """
    G = parabolic_gs_sor_matrix(N, r, omega)
    eigvals = np.linalg.eigvals(G)
    rho = np.max(np.abs(eigvals))
    return rho, eigvals


def plot_omega_vs_rho(
    N=32,
    xa=0.0,
    xb=1.0,
    cfl=0.8,
    v=1.0,
    omega_min=0.01,
    omega_max=2.0,
    domega=0.001,
    savepath="spec_1D_parabolic_gs_sor_N32.png",
):
    """
    Plot omega vs rho for the 1D parabolic GS-type SOR method.

    For parabolic scaling:
        dt = cfl * dx**2 / v

    so:
        r = v*dt/dx**2 = cfl
    """
    dx = (xb - xa) / N
    dt = cfl * dx**2 / (2.0 * v)
    r = v * dt / dx**2

    omega_values = np.arange(omega_min, omega_max + 0.5 * domega, domega)
    rho_values = np.zeros(len(omega_values))

    for j, omega in enumerate(omega_values):
        rho, _ = spectral_radius(N, r, omega)
        rho_values[j] = rho

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(omega_values, rho_values, linewidth=2)
    ax.axhline(1.0, linestyle="--", linewidth=1)

    ax.grid(True)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho(G)$")
    ax.set_title(
        rf"1D Parabolic GS-SOR Spectral Radius, "
        rf"$N={N}$, $r={r:.4f}$, CFL={cfl}"
    )

    plt.tight_layout()
    plt.savefig(savepath, dpi=400)
    plt.show()

    return {
        "N": N,
        "dx": dx,
        "dt": dt,
        "r": r,
        "omega_values": omega_values,
        "rho_values": rho_values,
    }


if __name__ == "__main__":
    results = plot_omega_vs_rho(
        N=32,
        xa=0.0,
        xb=1.0,
        cfl=0.8,
        v=0.5,
        omega_min=0.01,
        omega_max=3.0,
        domega=0.001,
        savepath="spec_1D_parabolic_gs_sor_N32.png",
    )