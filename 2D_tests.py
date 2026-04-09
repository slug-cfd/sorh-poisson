# perform n runs of unsplit method with theta randomized per run
def random_test(n, config):
    if config.method != "h_unsplit":
        return

    print("method:", config.method)
    print()

    iter_min = 1e7

    for _ in range(n):
        U, t, iter_count, histories = solve(config)

        res_final = histories["exact"][-1]
        conv_history = histories["exact"]

        if res_final < 1.0 and iter_count < 1000:
            print("final residual: ", res_final)
            print("iterations:", iter_count)
            if iter_count < iter_min:
                iter_min = iter_count
            # print("time t: ", t)
            # plot(U, iter_count, err_conv, params)
            print()

    print("iter_min:", iter_min)


def sweep_omega_all_methods(
    problem="poisson",
    methods=None,
    Nx=32,
    Ny=32,
    cfl=0.8,
    iter_max=4000,
    norm="fro",
    omega_min=0.05,
    omega_max=2.50,
    d_omega=0.05,
    res_threshold=1e3,
    outdir="results",
    outfile=None,
):
    """
    Sweep a common omega range across all methods and save run statistics.

    Stability / convergence logic:
      - stable     := max(exact residual history) <= res_threshold
      - convergent := stable and final exact residual <= res_threshold

    Assumes:
      solve(config) returns (U, t, iter_count, histories)
      histories["exact"] is the exact residual history
    """

    if methods is None:
        methods = ["p_unsplit", "p_ADI", "h_unsplit", "h_ADI"]

    if problem == "poisson":
        bc_type = "poisson"
        f = poisson2D.f
        phi_exact = poisson2D.phi_exact
    else:
        bc_type = "linear_in_x"
        f = poisson2D.f
        phi_exact = poisson2D.phi_exact

    # safer than np.arange for floating sweeps
    n_omegas = int(round((omega_max - omega_min) / d_omega)) + 1
    omegas = np.round(np.linspace(omega_min, omega_max, n_omegas), 12)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if outfile is None:
        # outfile = outdir / (
        #     f"omega_sweep_{problem}_Nx{Nx}_Ny{Ny}_"
        #     f"omin{omega_min:g}_omax{omega_max:g}_dom{d_omega:g}.csv"
        # )
        outfile = outdir / f"omega_Nx{Nx}_Ny{Ny}.csv"
    else:
        outfile = Path(outfile)

    rows = []

    print(f"Running omega sweep for methods: {methods}")
    print(f"omega range: [{omega_min}, {omega_max}] with d_omega = {d_omega}")
    print(f"res_threshold = {res_threshold}")
    print()

    for method in methods:
        print(f"=== method: {method} ===")

        for omega in omegas:
            print(f"=== omega: {omega} ===")
            config = config2D.Config(
                problem=problem,
                method=method,
                bc_type=bc_type,
                Nx=Nx,
                Ny=Ny,
                cfl=cfl,
                omega=float(omega),
                iter_max=iter_max,
                norm=norm,
                f=f,
                phi_exact=phi_exact,
            )

            start = time.perf_counter()

            try:
                U, t, iter_count, histories = solve(config)
                elapsed = time.perf_counter() - start

                exact_hist = np.asarray(histories["exact"], dtype=float)

                # guard against empty / invalid histories
                if exact_hist.size == 0 or np.any(~np.isfinite(exact_hist)):
                    res_min = np.nan
                    res_max = np.nan
                    res_final = np.nan
                    stable = False
                    convergent = False
                else:
                    res_min = float(np.min(exact_hist))
                    res_max = float(np.max(exact_hist))
                    res_final = float(exact_hist[-1])

                    stable = bool(res_max <= res_threshold)
                    convergent = bool(stable and res_final <= res_threshold)

                row = {
                    "method": method,
                    "omega": float(omega),
                    "iter_count": int(iter_count),
                    "res_min": res_min,
                    "res_max": res_max,
                    "res_final": res_final,
                    "elapsed_sec": float(elapsed),
                    "stable": int(stable),
                    "convergent": int(convergent),
                }

                rows.append(row)

                # print(
                #     f"omega={omega:6.3f} | "
                #     f"iters={iter_count:5d} | "
                #     f"res_final={res_final:12.5e} | "
                #     f"res_max={res_max:12.5e} | "
                #     f"time={elapsed:8.3f}s | "
                #     f"stable={stable}"
                # )

            except Exception as e:
                elapsed = time.perf_counter() - start

                row = {
                    "method": method,
                    "omega": float(omega),
                    "iter_count": -1,
                    "res_min": np.nan,
                    "res_max": np.nan,
                    "res_final": np.nan,
                    "elapsed_sec": float(elapsed),
                    "stable": 0,
                    "convergent": 0,
                    "error": str(e),
                }
                rows.append(row)

                # print(
                #     f"omega={omega:6.3f} | FAILED | "
                #     f"time={elapsed:8.3f}s | error={e}"
                # )

        print()

    # Save CSV manually so you do not need pandas
    headers = [
        "method",
        "omega",
        "iter_count",
        "res_min",
        "res_max",
        "res_final",
        "elapsed_sec",
        "stable",
        "convergent",
    ]

    # include error column only if any run failed
    if any("error" in row for row in rows):
        headers.append("error")

    with open(outfile, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            vals = []
            for h in headers:
                v = row.get(h, "")
                vals.append(str(v))
            f.write(",".join(vals) + "\n")

    print(f"Saved sweep data to: {outfile}")

    return rows


# run the randomized unsplit test
# random_test(100, config)

# plot phi_exact for Poisson test eqn
# plot_exact_solution(config)

# run the test over omega
# results = sweep_omega_all_methods(
#     problem=problem,
#     methods=methods,
#     Nx=Nx,
#     Ny=Ny,
#     cfl=cfl,
#     iter_max=iter_max,
#     norm=norm,
#     omega_min=0.05,
#     omega_max=2.50,
#     d_omega=0.05,
#     res_threshold=1e3,
#     outdir="results",
# )
