def nonlinear_v(u):
    return u / (1.0 + u)


def nonlinear_face_values(U, i):
    um = 0.5 * (U[i - 1] + U[i])
    up = 0.5 * (U[i] + U[i + 1])
    return nonlinear_v(um), nonlinear_v(up)
