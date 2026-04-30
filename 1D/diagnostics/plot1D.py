import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config1D import method_name_map
from problem_helpers import exact_diagnostic


def plot_solution(U, iter_count, histories, config):
    method_name = method_name_map[config.method]
    exact_error = np.abs(exact_diagnostic(U, config))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"1D Results: {method_name}", fontsize=14)

    axes[0].plot(config.x, U, linewidth=2)
    axes[0].set_title("Final Solution")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("U(x)")
    axes[0].grid(True)

    axes[1].semilogy(range(1, iter_count + 1), histories["exact"][:iter_count], linewidth=2)
    axes[1].set_title("Exact Error History")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Exact diagnostic norm")
    axes[1].grid(True)

    axes[2].plot(config.x, exact_error, linewidth=2)
    axes[2].set_title("Exact Diagnostic")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("Residual")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()
