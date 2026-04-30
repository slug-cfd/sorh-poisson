methods = {
    "poisson_base": ["p_unsplit", "p_ADI", "h_unsplit", "h_ADI"],

    "poisson_mg": ["p_unsplit", "p_ADI", "h_unsplit", "h_ADI"],

    "toy_nonlinear": ["p_unsplit", "h_unsplit", "DS1", "DS2"],

    "toy_linear": [],
}

### Old Notes ###
# h_unsplit (N=32, weighted hybrid): a = b = 0.8: 0.687
# p_unsplit (N=32, non-multigrid): omega 2.294
# p_unsplit multigrid vs. non-multigrid comparison (N=64):
# -> non-multigrid:     omega = 2.390   -> ~1.1s
# -> multigrid (fmg):   omega = 2.300   -> ~0.65s   -> almost 2x speedup!
### ~~~~~~~~~ ###

# each list of omegas corresponds to the respective method in `methods` above!

omegas32 = {    # Nx = Ny = 32
    "poisson_base": [2.000, 1.81, 0.687, 0.25],

    "poisson_mg": [2.000, 1.0, 0.670, 1.0],

    "toy_nonlinear": [1.55, 0.82, 0.80, 0.80],

    "toy_linear": [],
}

omegas64 = {    # Nx = Ny = 64
    "poisson_base": [2.000, 1.81, 0.713, 0.25],

    "poisson_mg": [2.000, 1.0, 0.670, 1.0],

    "toy_nonlinear": [1.55, 0.82, 0.80, 0.80],

    "toy_linear": [],
}

omegas128 = {    # Nx = Ny = 128
    "poisson_base": [2.457, 1.81, 0.731, 0.25],

    "poisson_mg": [2.000, 1.0, 0.670, 1.0],

    "toy_nonlinear": [1.55, 0.82, 0.80, 0.80],

    "toy_linear": [],
}
