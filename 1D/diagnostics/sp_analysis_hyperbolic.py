import numpy as np
from matplotlib import pyplot as plt


def hyperbolic_sor_matrix(N, r, omega):
    """
    N+1 total grid points: 0, ..., N
    Interior unknowns: 1, ..., N-1
    """
    m = N - 1

    L = np.eye(m)
    for i in range(1, m):
        L[i, i-1] = -omega

    R = np.zeros((m, m))
    for i in range(m):
        R[i, i] = (1.0 / omega) - (2.0 * r)
        if i > 0:
            R[i, i-1] = r - 1
        if i < m - 1:
            R[i, i+1] = r

    G = np.linalg.solve(L, omega * R)
    return G


def spectral_radius(N, r, omega):
    G = hyperbolic_sor_matrix(N, r, omega)
    eigvals = np.linalg.eigvals(G)
    return max(abs(eigvals)), eigvals


N = 32
xa, xb = 0.0, 1.0
dx = (xb - xa) / N
cfl = 0.8
v = 1.0

dt = cfl * (dx / v)
r = v * dt / dx

omega_values = np.arange(0.50, 1.50, 0.001)
stable = np.zeros(len(omega_values))

def plot_rho_vs_omega():
    for j, omega in enumerate(omega_values):
        rho, _ = spectral_radius(N, r, omega)
        stable[j] = rho

    # print(stable)

    fig, ax = plt.subplots()
    # ax.plot(omega_values, stable, marker='o', markersize=4)
    ax.plot(omega_values, stable, linewidth=2)

    ax.grid(True)
    ax.set_xlabel("omega")
    ax.set_ylabel("rho")
    # plt.savefig("spec_1D_N32.png", dpi=400)
    plt.show()


def show_dom_eigs():
    dom_eigs = []

    for omega in omega_values:
        rho, eigvals = spectral_radius(N, r, omega)
        lam = eigvals[np.argmax(np.abs(eigvals))]
        dom_eigs.append(lam)

    dom_eigs = np.array(dom_eigs)

    plt.figure()
    plt.plot(omega_values, np.real(dom_eigs), label="real")
    plt.plot(omega_values, np.imag(dom_eigs), label="imag")
    plt.grid(True)
    plt.xlabel("omega")
    plt.ylabel("dominant eigenvalue")
    plt.legend()
    # plt.savefig("dom_eig_1D_N32.png", dpi=400)
    plt.show()


def plot_eigvals():
    for omega in [0.8, 0.91, 1.0, 1.1]:
        rho, eigvals = spectral_radius(N, r, omega)

        plt.figure()
        plt.scatter(np.real(eigvals), np.imag(eigvals), s=12)

        theta = np.linspace(0, 2*np.pi, 400)
        plt.plot(np.cos(theta), np.sin(theta), linestyle="--")

        plt.gca().set_aspect("equal")
        plt.grid(True)
        plt.xlabel("Re(lambda)")
        plt.ylabel("Im(lambda)")
        plt.title(f"omega={omega:.3f}, rho={rho:.4f}")
        # plt.savefig(f"eigvals_omega_{omega:.3f}.png", dpi=400)
        plt.show()


def make_stability_map(
    N=32,
    r_min=0.01,
    r_max=2.0,
    dr=0.01,
    omega_min=0.01,
    omega_max=1.99,
    domega=0.01,
    savepath="stability_map_1D.png",
):
    r_values = np.arange(r_min, r_max + 0.5*dr, dr)
    omega_values = np.arange(omega_min, omega_max + 0.5*domega, domega)
    rho_map = np.zeros((len(r_values), len(omega_values)))

    for i, r in enumerate(r_values):
        for j, omega in enumerate(omega_values):
            rho, _ = spectral_radius(N, r, omega)
            rho_map[i, j] = rho

    R, W = np.meshgrid(omega_values, r_values)

    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.contourf(W, R, rho_map, levels=100)
    fig.colorbar(c, ax=ax, label=r"$\rho(G)$")

    # Stability boundary rho = 1
    ax.contour(W, R, rho_map, levels=[1.0], linewidths=2)

    # Optional: stable region overlay
    ax.contourf(
        W, R, rho_map < 1.0,
        levels=[0.5, 1.5],
        alpha=0.15,
    )

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$r = v\Delta t/\Delta x$")
    ax.set_title(f"1D Hyperbolic SOR Stability Map, N={N}")
    ax.grid(True)
    plt.tight_layout()
    # plt.savefig(savepath, dpi=400)
    plt.show()

    return {
        "r_values": r_values,
        "omega_values": omega_values,
        "rho_map": rho_map,
        "stable_mask": rho_map < 1.0,
    }

# results = make_stability_map(
#     N=32,
#     r_min=0.01,
#     r_max=2.0,
#     dr=0.005,
#     omega_min=0.50,
#     omega_max=1.50,
#     domega=0.005,
#     savepath="stability_map_1D_N32.png",
# )

if __name__ == "__main__":
    plot_rho_vs_omega()
