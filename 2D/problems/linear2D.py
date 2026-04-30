from methods import linear as lin

def solve(params):
    match params.method:
        case "parabolic_jacobi":
            U, t, iter_count, res, err_conv = lin.parabolic(params, iter_method="jacobi")
        case "parabolic_gs":
            U, t, iter_count, res, err_conv = lin.parabolic(params, iter_method="gs")
        case "parabolic_sor":
            U, t, iter_count, res, err_conv = lin.parabolic(params, iter_method="sor")
        # case "DS1":     # 1st-order dimensional splitting (DS1)
        #     U, t, iter_count, res, err_conv = lin.DS1(params)
        # case "DS2":     # 2nd-order dimensional splitting (DS2)
        #     U, t, iter_count, res, err_conv = lin.DS2(params)
        # case "unsplit":
        #     U, t, iter_count, res, err_conv = lin.unsplit(params)
        case _:
            return -1

    # return U, t, iter_count, res, err_conv, err_exact
    return U, t, iter_count, res, err_conv
