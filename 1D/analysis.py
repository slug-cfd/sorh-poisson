# analysis.py: analysis for 1D methods

import matplotlib.pyplot as plt
import numpy as np
# from scipy.optimize import curve_fit


def plot_omega_vs_iter_count(N):
    methods = ["sornl", "sorh"]
    test_param = "omega"
    comp_metric = "iter_count"

    # Ns = [32, 64]

    filenames = [f"map_{method}_{test_param}_vs_{comp_metric}_n{N}.dat" for method in methods]
    # filenames = [f"map_sorh_{test_param}_vs_{comp_metric}_n{N}.dat" for N in Ns]

    max_iter = 10000

    data1 = np.loadtxt(filenames[0])
    data2 = np.loadtxt(filenames[1])

    x1, y1 = data1[:, 0], data1[:, 1]
    x2, y2 = data2[:, 0], data2[:, 1]

    # find cutoffs independently
    idx1 = np.argmax(y1 < max_iter)
    idx2 = np.argmax(y2 < max_iter)

    x1_plot, y1_plot = x1[idx1:], y1[idx1:]
    x2_plot, y2_plot = x2[idx2:], y2[idx2:]

    plt.plot(x1_plot, y1_plot, label=methods[0])
    plt.plot(x2_plot, y2_plot, label=methods[1])
    # plt.plot(x1_plot, y1_plot, label=f"N={Ns[0]}")
    # plt.plot(x2_plot, y2_plot, label=f"N={Ns[1]}")
    plt.xlabel("omega")
    plt.ylabel("iteration count")
    plt.grid(which='both')
    plt.legend()
    plt.show()


def plot_omegas_vs_n():
    method = "sorh"
    filename = f"{method}_optimal_omega_vs_N.dat"

    data = np.loadtxt(filename)
    Ns, omegas, iter_counts = data[:, 0], data[:, 1], data[:, 2]

    # plt.plot(Ns, omegas)
    # plt.xlabel("N")
    # plt.ylabel("omega")
    # plt.grid(which='both')
    # plt.show()

    plt.plot(Ns, iter_counts)
    plt.xlabel("N")
    plt.ylabel("iteration count")
    plt.grid(which='both')
    plt.show()


# plot_omegas_vs_n()
plot_omega_vs_iter_count(32)
