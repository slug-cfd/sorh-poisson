import numpy as np


def exact_diagnostic(U, config):
    if config.problem == "linear":
        return U - config.U_exact

    return (U - np.log1p(U)) - (1.0 - np.log(2.0)) * config.x
