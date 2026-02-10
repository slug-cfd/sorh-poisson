import numpy as np

def update_BCs(U, x, bc_type):
    a, b, c, d = 1.0, 2.0, 1.0, 2.0

    # update BCs
    match bc_type:
        case 0:
            # BC: U(x, y) = x
            linear_in_x(U, x)
        case 1:
            # BC: spatial oscillations
            params = [a, b, c, d]
            sinusoidal(U, x, params)
        case 2:
            # BC: combination (mixed nonlinear functions)
            params = [a, b]
            combination(U, x, params)


# solution profile: U(x, y) = x
def linear_in_x(U, x):
    # x=0 -> u=0
    U[0, :] = 0.0
    # x=1 -> u=1
    U[-1, :] = 1.0
    # y=0 -> u=x
    U[:, 0] = x
    # y=1 -> u=x
    U[:, -1] = x

    return U

def sinusoidal(U, x, params):
    a, b, c, d = params

    U[:, 0] = a * np.sin(b * np.pi * x)
    U[:, -1] = 0.0
    U[0, :] = c * np.sin(d * np.pi * x)
    U[-1, :] = 0.0

    return U


def combination(U, x, params):
    a, b = params

    U[:, 0] = a * np.sin(b * np.pi * x)
    U[:, -1] = 0.0
    U[0, :] = -np.sqrt(x)
    U[-1, :] = np.sqrt(x)

    return U
