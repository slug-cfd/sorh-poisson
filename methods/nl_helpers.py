import numpy as np

def get_vx(U, params):
    Nx, Ny = params.Nx, params.Ny
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je

    vx = np.zeros((Nx, Ny, 2), dtype=np.float64)

    for j in range(jb, je + 1):
        for i in range(ib, ie + 1):
            # update values for v
            um = 0.5 * (U[i - 1, j] + U[i, j])
            up = 0.5 * (U[i, j] + U[i + 1, j])
            vm = um / (1.0 + um)
            vp = up / (1.0 + up)
            vx[i - 1, j - 1, 0] = vm
            vx[i - 1, j - 1, 1] = vp

    return vx


def get_vy(U, params):
    Nx, Ny = params.Nx, params.Ny
    ib, ie, jb, je = params.ib, params.ie, params.jb, params.je

    vy = np.zeros((Nx, Ny, 2), dtype=np.float64)

    for i in range(ib, ie + 1):
        for j in range(jb, je + 1):
            # update values for v
            um = 0.5 * (U[i, j - 1] + U[i, j])
            up = 0.5 * (U[i, j] + U[i, j + 1])
            vm = um / (1.0 + um)
            vp = up / (1.0 + up)
            vy[i - 1, j - 1, 0] = vm
            vy[i - 1, j - 1, 1] = vp

    return vy
