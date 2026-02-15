from methods import nonlinear as nl

def solve(params):
    match params.method:
        case "parabolic":
            U, t, iter_count, res, err_conv = nl.parabolic(params)
        case "DS1":     # 1st-order dimensional splitting (DS1)
            U, t, iter_count, res, err_conv = nl.DS1(params)
        case "DS2":     # 2nd-order dimensional splitting (DS2)
            U, t, iter_count, res, err_conv = nl.DS2(params)
        case "unsplit":
            U, t, iter_count, res, err_conv = nl.unsplit(params)
        case _:
            return -1

    # return U, t, iter_count, res, err_conv, err_exact
    return U, t, iter_count, res, err_conv
