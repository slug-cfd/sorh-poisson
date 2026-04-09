from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np

@dataclass
class Config:
    # — user‐set —
    problem:    str     # 'poisson', 'toy_linear', 'toy_nonlinear'
    method:     str     # 'p_unsplit', 'h_unsplit', 'p_ADI', 'h_ADI', etc.
    bc_type:    str     # 'poisson', 'linear_in_x', etc.

    Nx:         int     # number of INTERIOR cells (x-direction)
    Ny:         int     # number of INTERIOR cells (y-direction)
    cfl:        float   # CFL number

    omega:      float   # SOR over/under-relaxation parameter
    iter_max:   int     # maximum allowed iterations

    norm:       Any     # norm type. options: [1, 2, "fro", np.inf]

    f:          Callable = None     # RHS definition
    phi_exact:  Callable = None     # function for exact solution calculation

    mg_solver:  str = None      # multigrid algorithm to use (None -> no multigrid)

    # — derived —
    xa:         float   = field(init=False)  # left physical boundary (x)
    xb:         float   = field(init=False)  # right physical boundary (x)
    ya:         float   = field(init=False)  # left physical boundary (y)
    yb:         float   = field(init=False)  # right physical boundary (y)

    dx:         float   = field(init=False)
    dy:         float   = field(init=False)

    ngc:        int     = field(init=False)  # number of ghost cells per side
    ib:         int     = field(init=False)  # starting interior index (x)
    ie:         int     = field(init=False)  # ending interior index (x)
    jb:         int     = field(init=False)  # starting interior index (y)
    je:         int     = field(init=False)  # ending interior index (y)

    # Poisson-specific parameters
    R:          float = field(init=False)
    rho0:       float = field(init=False)
    G:          float = field(init=False)

    # — initial state arrays —
    x:          np.ndarray = field(init=False)  # includes ghosts
    y:          np.ndarray = field(init=False)  # "        "

    U0:         np.ndarray = field(init=False)  # initial state array

    f_all:      np.ndarray = field(init=False)  # exact RHS
    U_exact:    np.ndarray = field(init=False)  # exact solution (for Poisson problem)


    def __post_init__(self):
        # 1st quadrant only for Poisson
        self.xa = 0.0
        self.ya = 0.0

        if self.problem == "poisson":
            self.ngc = 1

            self.R = 1.0
            self.rho0 = 3.0
            self.G = 0.25 / np.pi   # 4pi*G = 1

            # all 4 quadrants
            self.xa = -20.0 * self.R
            self.ya = -20.0 * self.R

            self.xb = 20.0 * self.R
            self.yb = 20.0 * self.R
        else:
            self.ngc = 1

            self.xb = 1.0
            self.yb = 1.0

        # self.dx = self.xb / self.Nx
        # self.dy = self.yb / self.Ny

        self.ib = self.ngc
        self.jb = self.ngc
        # self.ie = self.Nx + self.ngc - 1
        # self.je = self.Ny + self.ngc - 1
        self.ie = self.Nx - self.ngc - 1
        self.je = self.Ny - self.ngc - 1


        # np.linspace includes the endpoints; these will also
        # automatically include ghost cells depending on self.ngc
        # self.x = np.linspace(self.xa - (self.dx * self.ngc),
        #                      self.xb + (self.dx * self.ngc), self.Nx + (2 * self.ngc))
        # self.y = np.linspace(self.ya - (self.dy * self.ngc),
        #                      self.yb + (self.dy * self.ngc), self.Ny + (2 * self.ngc))

        # np.linspace includes the endpoints; these will also
        # automatically include ghost cells depending on self.ngc
        self.x = np.linspace(self.xa, self.xb, self.Nx)
        self.y = np.linspace(self.ya, self.yb, self.Ny)

        self.dx = (self.xb - self.xa) / (self.Nx - 1)
        self.dy = (self.yb - self.ya) / (self.Ny - 1)

        # initialize U0
        # self.U0 = np.zeros((self.Nx + (2 * self.ngc), self.Ny + (2 * self.ngc)))
        self.U0 = np.zeros((self.Nx, self.Ny))

        if self.f:
            # apply f(x, y) over the whole 2D grid
            X, Y = np.meshgrid(self.x, self.y, indexing='ij')  # shape (Nx+2*ngc, Ny+2*ngc)
            self.f_all = self.f(X, Y, self.rho0, self.R)

        if self.phi_exact:
            # set the exact solution on full grid
            X, Y = np.meshgrid(self.x, self.y, indexing='ij')
            self.U_exact = self.phi_exact(X, Y, self.rho0, self.R, self.G)


