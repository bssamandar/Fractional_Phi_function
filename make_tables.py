# -*- coding: utf-8 -*-
"""Rows of Tables 1-8 of the manuscript (LaTeX), produced from results/*.json.

Each file tables/tab_*.tex contains only the body rows of the corresponding
table (between \\midrule and \\bottomrule).
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
TAB = os.path.join(ROOT, "tables")
os.makedirs(TAB, exist_ok=True)


def load(name):
    with open(os.path.join(RES, f"{name}.json")) as fh:
        return json.load(fh)


def write(name, rows):
    with open(os.path.join(TAB, name), "w") as fh:
        fh.write("\n".join(rows) + "\n")


def e(x):
    """number in the form 1.23e-4 (LaTeX math)"""
    if x < 1e-30:
        return "$<10^{-30}$"
    m, p = f"{x:.2e}".split("e")
    return f"${m}\\mathrm{{e}}{{{int(p)}}}$"


def eoc(a, b):
    """experimental order of convergence for doubling n"""
    return "--" if min(a, b) < 1e-30 else f"{math.log2(a / b):.2f}"


def main():
    # model integrands (Tables 2 and 3)
    r1 = load("exp1")
    ns = r1["0.1"]["f1"]["n"]
    keep = [i for i, n in enumerate(ns) if n >= 8]

    def block(d, i):
        out = []
        for m in ["opt", "trap"]:
            out.append(e(d[m][i]))
            out.append("--" if i == keep[0] else eoc(d[m][i - 1], d[m][i]))
        return out

    rows = []
    for i in keep:
        line = [str(ns[i])]
        for al in ["0.1", "0.5", "0.9"]:
            line += block(r1[al]["f1"], i)
        rows.append(" & ".join(line) + r" \\")
    write("tab_f1.tex", rows)

    rows = []
    for i in keep:
        line = [str(ns[i])]
        for fn in ["f2", "f3", "f4"]:
            line += block(r1["0.5"][fn], i)
        rows.append(" & ".join(line) + r" \\")
    write("tab_f234.tex", rows)

    # Gauss-Jacobi comparison (Table 4)
    rows = []
    for m, v in r1["gj"].items():
        gj = e(v[1]) if v[1] > 1.1e-40 else "$<10^{-39}$"
        rows.append(f"{m} & {e(v[0])} & {gj} \\\\")
    write("tab_gj.tex", rows)

    # norm of the error functional (Table 5)
    r2 = load("exp2")
    rows = []
    for i, n in enumerate(r2["0.1"]["n"]):
        line = [str(n)]
        for al in ["0.1", "0.5", "0.9"]:
            d = r2[al]["unif"]
            line += [e(d[i]), "--" if i == 0 else eoc(d[i - 1], d[i])]
        for al in ["0.1", "0.3"]:
            d = r2[al]["grad"]
            line += [e(d[i]), "--" if i == 0 else eoc(d[i - 1], d[i])]
        rows.append(" & ".join(line) + r" \\")
    write("tab_norm.tex", rows)

    # forward Abel transform and Abel inversion (Tables 6 and 7)
    r3, r4 = load("exp3"), load("exp4")
    rows = []
    for i, n in enumerate(r3["gauss"]["n"]):
        line = [str(n)]
        for p in ["gauss", "hollow", "cos"]:
            line += [e(r3[p]["opt"][i]), e(r3[p]["trap"][i])]
        line += [e(r3["gauss"]["opt_lam"][i])]
        rows.append(" & ".join(line) + r" \\")
    write("tab_abel_fwd.tex", rows)

    rows = []
    for i, n in enumerate(r4["gauss"]["n"]):
        line = [str(n)]
        for p in ["gauss", "hollow", "cos"]:
            line += [e(r4[p]["opt"][i]), e(r4[p]["trap"][i])]
        line += [e(r4["hollow"]["opt_noisy"][i]), e(r4["hollow"]["trap_noisy"][i])]
        rows.append(" & ".join(line) + r" \\")
    write("tab_abel_inv.tex", rows)

    # fractional relaxation (Table 8)
    r5 = load("exp5")
    rows = []
    for i, n in enumerate(r5["0.3"]["unif"]["n"]):
        line = [str(n)]
        for a in ["0.3", "0.5", "0.8"]:
            u, g = r5[a]["unif"], r5[a]["grad"]
            line += [e(u["opt"][i]), e(g["opt"][i]),
                     "--" if i == 0 else eoc(g["opt"][i - 1], g["opt"][i]), e(g["trap"][i])]
        rows.append(" & ".join(line) + r" \\")
    write("tab_relax.tex", rows)

    # cost and round-off of Algorithm 1 (Table 1)
    r6 = load("exp6")
    rows = []
    for i, n in enumerate(r6["n"]):
        rows.append(f"{n} & {r6['time_stable'][i] * 1e3:.1f} & {e(r6['err_stable'][i])} & "
                    f"{e(r6['err_stable_graded'][i])} & {e(r6['err_closed64'][i])} \\\\")
    write("tab_cost.tex", rows)
    print("tables written to", TAB)


if __name__ == "__main__":
    main()
