# -*- coding: utf-8 -*-
"""Write the raw numerical results (results/*.json) as CSV files in results/csv/."""
import csv
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "csv")
os.makedirs(OUT, exist_ok=True)


def load(name):
    with open(os.path.join(RES, f"{name}.json")) as fh:
        return json.load(fh)


def write(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main():
    r1 = load("exp1")
    rows = []
    for al, d in r1.items():
        if al == "gj":
            continue
        for fn, v in d.items():
            for n, eo, et in zip(v["n"], v["opt"], v["trap"]):
                rows.append([al, fn, n, repr(eo), repr(et)])
    write("exp1_model_integrands.csv", ["alpha", "integrand", "n", "abs_err_optimal", "abs_err_trapezoid"], rows)
    write("exp1_gauss_jacobi.csv", ["m_function_values", "abs_err_optimal", "abs_err_gauss_jacobi"],
          [[m, repr(v[0]), repr(v[1])] for m, v in r1["gj"].items()])

    r2 = load("exp2")
    rows = []
    for al, d in r2.items():
        for i, n in enumerate(d["n"]):
            rows.append([al, n, repr(d["unif"][i]), repr(d["bound_unif"][i]), d["r"], repr(d["grad"][i]),
                         repr(d["limit_grad"])])
    write("exp2_error_norm.csv", ["alpha", "n", "norm_uniform", "upper_bound_uniform", "r_graded",
                                  "norm_graded", "limit_n_times_norm_graded"], rows)

    r3 = load("exp3")
    rows = []
    for p, d in r3.items():
        for i, n in enumerate(d["n"]):
            rows.append([p, n, repr(d["opt"][i]), repr(d["trap"][i]), repr(d["opt_lam"][i])])
    write("exp3_abel_forward.csv", ["profile", "n", "max_err_optimal", "max_err_trapezoid",
                                    "max_err_optimal_lambda"], rows)

    r4 = load("exp4")
    rows = []
    for p, d in r4.items():
        if p == "profile":
            continue
        for i, n in enumerate(d["n"]):
            rows.append([p, n, repr(d["opt"][i]), repr(d["trap"][i]), repr(d["opt_noisy"][i]),
                         repr(d["trap_noisy"][i])])
    write("exp4_abel_inversion.csv", ["profile", "n", "max_err_optimal", "max_err_trapezoid",
                                      "max_err_optimal_noisy", "max_err_trapezoid_noisy"], rows)
    pr = r4["profile"]
    write("exp4_hollow_profile_n64.csv", ["r", "eps_exact", "eps_recovered_optimal", "y", "F"],
          [[repr(a), repr(b), repr(c), repr(d), repr(e)]
           for a, b, c, d, e in zip(pr["r"], pr["true"], pr["opt"], pr["y"], pr["F"])])

    r5 = load("exp5")
    rows = []
    for al, d in r5.items():
        for mesh, v in d.items():
            for i, n in enumerate(v["n"]):
                rows.append([al, mesh, v["r"], n, repr(v["opt"][i]), repr(v["trap"][i])])
    write("exp5_relaxation.csv", ["alpha", "mesh", "r", "n", "max_err_optimal", "max_err_trapezoid"], rows)

    r6 = load("exp6")
    write("exp6_cost_roundoff.csv", ["n", "time_s", "rel_err_alg1_uniform", "rel_err_alg1_graded",
                                     "rel_err_closed_form_uniform"],
          [[n, repr(r6["time_stable"][i]), repr(r6["err_stable"][i]), repr(r6["err_stable_graded"][i]),
            repr(r6["err_closed64"][i])] for i, n in enumerate(r6["n"])])
    print("CSV files written to", OUT)


if __name__ == "__main__":
    main()
