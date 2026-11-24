# -*- coding: utf-8 -*-
"""
Numerical experiments of the manuscript.

 Exp.1  accuracy & convergence for model integrands (mpmath, 40 digits)
 Exp.2  worst-case error ||R|| on uniform / graded meshes
 Exp.3  forward Abel transform (plasma emission tomography)
 Exp.4  Abel inversion = first-kind Abel integral equation
 Exp.5  second-kind Abel equation: fractional relaxation, graded meshes
 Exp.6  cost / round-off of the stable float64 algorithm
Results -> results/*.json, figures -> figures/*.pdf
"""
import json, os, time
import numpy as np
import mpmath as mp
from optquad import *

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#222222", "#6b6b6b"

def save(name, obj):
    with open(os.path.join(RES, f"{name}.json"), "w") as fh:
        json.dump(obj, fh, indent=1, default=float)

def _load(name):
    with open(os.path.join(RES, f"{name}.json")) as fh:
        return json.load(fh)

def rate(e):
    return [None] + [float(np.log2(e[i - 1] / e[i])) for i in range(1, len(e))]

# ---------------------------------------------------------------------------
# Exp. 1 : model integrands, uniform meshes, alpha in {0.1,0.5,0.9}
# ---------------------------------------------------------------------------
def exp1():
    mp.mp.dps = 40
    t = mp.mpf(1); b = 1 + 1j
    tests = {
        "f1": (lambda x: mp.sin(x) * mp.exp(x), lambda a: mp.im(I_exp(b, t, a))),
        "f2": (lambda x: mp.cos(4 * x), lambda a: mp.re(I_exp(4j, t, a))),
        "f3": (lambda x: x**mp.mpf('0.7'), lambda a: t**(a + mp.mpf('0.7')) * mp.beta(a, mp.mpf('1.7'))),
        "f4": (lambda x: mp.cosh(x), lambda a: (I_exp(1, t, a).real + I_exp(-1, t, a).real) / 2),
    }
    ns = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    res = {}
    for al in ['0.1', '0.5', '0.9']:
        a = mp.mpf(al); res[al] = {k: {"n": ns, "opt": [], "trap": []} for k in tests}
        for n in ns:
            X = mesh_uniform(t, n); Wo, Wt = weights_opt(X, t, a), weights_trap(X, t, a)
            for name, (f, ex) in tests.items():
                Iex = ex(a); fx = [f(x) for x in X]
                res[al][name]["opt"].append(float(abs(Iex - mp.fsum(w * v for w, v in zip(Wo, fx)))))
                res[al][name]["trap"].append(float(abs(Iex - mp.fsum(w * v for w, v in zip(Wt, fx)))))
    # Gauss-Jacobi vs optimal at equal number of f-evaluations
    gj = {}
    a = mp.mpf('0.1'); f, ex = tests["f1"]; Iex = ex(a)
    for m in [3, 5, 9, 13, 17, 33]:
        X = mesh_uniform(t, m - 1)
        gj[m] = (float(abs(Iex - apply(weights_opt(X, t, a), X, f))),
                 float(max(abs(Iex - quad_gj(f, t, a, m)), mp.mpf(10)**-40)))
    res["gj"] = gj
    save("exp1", res)
    return res

# ---------------------------------------------------------------------------
# Exp. 2 : norm of the error functional
# ---------------------------------------------------------------------------
def exp2():
    mp.mp.dps = 30
    t = mp.mpf(1)
    ns = [8, 16, 32, 64, 128, 256, 512, 1024]
    res = {}
    for al in ['0.1', '0.3', '0.5', '0.7', '0.9']:
        a = mp.mpf(al)
        res[al] = {"n": ns,
                   "unif": [float(error_norm(mesh_uniform(t, n), t, a)) for n in ns],
                   "bound_unif": [float(norm_upper_bound(mesh_uniform(t, n), t, a)) for n in ns]}
        r = 3 / (2 * a + 1)          # asymptotically optimal grading exponent (Theorem on asymptotics)
        res[al]["r"] = float(r)
        res[al]["grad"] = [float(error_norm(mesh_graded(t, n, r), t, a)) for n in ns]
        res[al]["limit_grad"] = float(mp.mpf(3) / 2 * (2 * a + 1)**(-mp.mpf(3) / 2) * t**(a + mp.mpf(1) / 2))
    save("exp2", res)
    return res

# ---------------------------------------------------------------------------
# Exp. 3 : forward Abel transform  F(y) = 2 int_y^1 eps(r) r dr / sqrt(r^2-y^2)
#          s = 1 - r^2, tau = 1 - y^2 :  F = int_0^tau (tau-s)^(-1/2) eps(sqrt(1-s)) ds
# ---------------------------------------------------------------------------
SIG = mp.mpf('0.5')
def eps_gauss(r): return mp.exp(-r**2 / SIG**2)
def eps_hollow(r): return (1 + 3 * r**2) * mp.exp(-r**2 / SIG**2)      # "hollow" profile
def eps_cos(r): return mp.cos(mp.pi * r / 2)**2

def F_exact(eps, tau):
    tau = mp.mpf(tau)
    if tau == 0:
        return mp.mpf(0)
    y2 = 1 - tau
    return 2 * mp.quad(lambda v: eps(mp.sqrt(y2 + v**2)), [0, mp.sqrt(tau)])

def exp3():
    mp.mp.dps = 30
    a = 0.5
    lam = float(1 / SIG**2)
    ns = [8, 16, 32, 64, 128, 256, 512]
    profiles = {"gauss": eps_gauss, "hollow": eps_hollow, "cos": eps_cos}
    wf = {"opt": lambda X, t: weights_opt_stable(X, t, a),
          "opt_lam": lambda X, t: lam**(-a) * weights_opt_stable(lam * X, lam * t, a),
          "trap": lambda X, t: weights_trap_stable(X, t, a)}
    res = {}
    for pname, eps in profiles.items():
        res[pname] = {"n": ns, "opt": [], "opt_lam": [], "trap": []}
        for n in ns:
            Xm = mesh_uniform(1, n); X = np.array([float(x) for x in Xm])
            fx = np.array([float(eps(mp.sqrt(1 - x))) for x in Xm])
            Fex = np.array([float(F_exact(eps, x)) for x in Xm])
            for key, W in wf.items():
                err = max(abs(np.dot(W(X[:m + 1], X[m]), fx[:m + 1]) - Fex[m]) for m in range(1, n + 1))
                res[pname][key].append(float(err))
        print("exp3", pname, res[pname])
    save("exp3", res)
    return res

# ---------------------------------------------------------------------------
# Exp. 4 : Abel inversion (first-kind Abel integral equation)
#   given F(tau_m) on the grid, recover eps via f_m, f_0 = eps(1) known
# ---------------------------------------------------------------------------
def abel_invert(Fm, X, a, wfun, f0):
    n = len(X) - 1
    f = np.zeros(n + 1); f[0] = f0
    for m in range(1, n + 1):
        W = wfun(X[:m + 1], X[m], a)
        f[m] = (Fm[m] - np.dot(W[:m], f[:m])) / W[m]
    return f

def exp4():
    mp.mp.dps = 30
    a = 0.5
    ns = [16, 32, 64, 128, 256, 512]
    res = {}
    rng = np.random.default_rng(2026)
    for pname, eps in {"gauss": eps_gauss, "hollow": eps_hollow, "cos": eps_cos}.items():
        res[pname] = {"n": ns, "opt": [], "trap": [], "opt_noisy": [], "trap_noisy": []}
        for n in ns:
            Xm = mesh_uniform(1, n); X = np.array([float(x) for x in Xm])
            Fm = np.array([float(F_exact(eps, x)) for x in Xm])
            ftrue = np.array([float(eps(mp.sqrt(1 - x))) for x in Xm])
            noise = 1e-4 * np.max(np.abs(Fm)) * rng.standard_normal(n + 1)
            for key, wf in [("opt", weights_opt_stable), ("trap", weights_trap_stable)]:
                fr = abel_invert(Fm, X, a, wf, ftrue[0])
                res[pname][key].append(float(np.max(np.abs(fr - ftrue))))
                fr = abel_invert(Fm + noise, X, a, wf, ftrue[0])
                res[pname][key + "_noisy"].append(float(np.max(np.abs(fr - ftrue))))
        print("exp4", pname, res[pname])
    # one profile for the figure
    n = 64; Xm = mesh_uniform(1, n); X = np.array([float(x) for x in Xm])
    Fm = np.array([float(F_exact(eps_hollow, x)) for x in Xm])
    ftrue = np.array([float(eps_hollow(mp.sqrt(1 - x))) for x in Xm])
    fr = abel_invert(Fm, X, a, weights_opt_stable, ftrue[0])
    res["profile"] = {"r": list(np.sqrt(1 - X)), "true": list(ftrue), "opt": list(fr),
                      "F": list(Fm), "y": list(np.sqrt(1 - X))}
    save("exp4", res)
    return res

# ---------------------------------------------------------------------------
# Exp. 5 : fractional relaxation  y(t) = 1 - kappa I^alpha y(t),  y = E_alpha(-kappa t^alpha)
#          (equivalently  C D^alpha y = -kappa y, y(0)=1)
# ---------------------------------------------------------------------------
def ml(a, z):
    with mp.workdps(60):
        a = mp.mpf(a); z = mp.mpf(z)
        return mp.nsum(lambda k: z**k / mp.gamma(a * k + 1), [0, mp.inf])

def relax(X, a, kappa, wfun):
    n = len(X) - 1; y = np.zeros(n + 1); y[0] = 1.0
    c = kappa / float(mp.gamma(a))
    for m in range(1, n + 1):
        W = wfun(X[:m + 1], X[m], a)
        y[m] = (1 - c * np.dot(W[:m], y[:m])) / (1 + c * W[m])
    return y

def exp5():
    T, kappa = 1.0, 1.0
    ns = [16, 32, 64, 128, 256, 512, 1024]
    res = {}
    for a in [0.3, 0.5, 0.8]:
        res[str(a)] = {}
        for mesh in ["unif", "grad"]:
            r = 1.0 if mesh == "unif" else 2.0 / a
            eo, et = [], []
            for n in ns:
                X = T * (np.arange(n + 1) / n)**r
                yex = np.array([float(ml(a, -kappa * x**a)) for x in X])
                eo.append(float(np.max(np.abs(relax(X, a, kappa, weights_opt_stable) - yex))))
                et.append(float(np.max(np.abs(relax(X, a, kappa, weights_trap_stable) - yex))))
            res[str(a)][mesh] = {"n": ns, "r": r, "opt": eo, "trap": et}
        print("exp5", a, res[str(a)])
    save("exp5", res)
    return res

# ---------------------------------------------------------------------------
# Exp. 6 : cost and round-off of the float64 algorithm
# ---------------------------------------------------------------------------
def exp6():
    mp.mp.dps = 60
    res = {"n": [], "time_stable": [], "err_stable": [], "err_closed64": [], "err_stable_graded": []}
    a = 0.1; t = 1.0
    r = 3 / (2 * a + 1)
    weights_opt_stable(np.linspace(0, t, 11), t, a)          # warm-up (Gauss rules)
    for n in [100, 1000, 10000, 100000]:
        X = np.linspace(0, t, n + 1)
        reps = 20
        t0 = time.perf_counter()
        for _ in range(reps):
            A = weights_opt_stable(X, t, a)
        dt = (time.perf_counter() - t0) / reps
        A2 = weights_opt_fast(X, t, a)
        # weights compared with the reference: 21 equally spaced indices and the ten
        # weights next to each end-point (A_k depends only on x_{k-1}, x_k, x_{k+1})
        idx = np.unique(np.concatenate([np.linspace(0, n, 21).astype(int), np.arange(0, 10), np.arange(n - 9, n + 1)]))
        Xm = [mp.mpf(x) for x in X]
        es, ec = 0, 0
        for k in idx:
            ref = float(local_weight(Xm, k, t, a))
            es = max(es, abs(A[k] - ref) / abs(ref))
            ec = max(ec, abs(A2[k] - ref) / abs(ref))
        # mesh graded towards the singular end-point x = t
        Xg = t - t * (1 - np.arange(n + 1) / n)**r
        Xg[0], Xg[-1] = 0.0, t
        Ag = weights_opt_stable(Xg, t, a)
        Xgm = [mp.mpf(x) for x in Xg]
        eg = 0
        for k in idx:
            ref = float(local_weight(Xgm, k, t, a))
            eg = max(eg, abs(Ag[k] - ref) / abs(ref))
        res["n"].append(n); res["time_stable"].append(dt)
        res["err_stable"].append(es); res["err_closed64"].append(ec); res["err_stable_graded"].append(eg)
        print("exp6", n, dt, es, ec, eg)
    save("exp6", res)
    return res

def local_weight(X, k, t, a):
    """A_k from the two adjacent intervals only (mpmath)"""
    t = mp.mpf(t); a = mp.mpf(a); n = len(X) - 1
    def PQ(L, R):
        uL, uR = t - L, t - R
        dG = lgam(a, uL) - lgam(a, uR); dE = Eplus(a, uL) - Eplus(a, uR)
        P = (mp.exp(uL) * dG - mp.exp(-uL) * dE) / 2
        Q = (mp.exp(-uR) * dE - mp.exp(uR) * dG) / 2
        return P / mp.sinh(R - L), Q / mp.sinh(R - L)
    s = mp.mpf(0)
    if k > 0:
        s += PQ(X[k - 1], X[k])[0]
    if k < n:
        s += PQ(X[k], X[k + 1])[1]
    return s

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _legend(ax, color_items, style=True, **kw):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=c, lw=1.6, label=l) for c, l in color_items]
    if style:
        h += [Line2D([], [], color=INK, marker="o", lw=1.2, ms=4, label="optimal"),
              Line2D([], [], color=INK, marker="s", ls="--", mfc="white", lw=1.2, ms=4, label="trapezoid")]
    ax.legend(handles=h, **kw)


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": .6,
                         "lines.linewidth": 1.6, "lines.markersize": 4.5, "legend.frameon": False})
    cols = {"0.1": C1, "0.5": C2, "0.9": C3, "0.3": C4, "0.7": "#e87ba4"}
    r1, r2 = _load("exp1"), _load("exp2")

    # Fig 1: ||R|| uniform vs graded
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for al in ['0.1', '0.5', '0.9']:
        d = r2[al]
        ax[0].loglog(d["n"], d["unif"], "o-", c=cols[al], label=fr"$\alpha={al}$")
        ax[0].loglog(d["n"], d["bound_unif"], ":", c=cols[al])
        ax[1].loglog(d["n"], d["unif"], "o-", c=cols[al], label=fr"uniform, $\alpha={al}$")
        ax[1].loglog(d["n"], d["grad"], "s--", c=cols[al], mfc="white", label=fr"graded, $r={d['r']:.2f}$")
    n = np.array(r2['0.1']["n"], float)
    ax[0].loglog(n, 0.9 * n**-0.6, c=MUTED, lw=.8); ax[0].text(300, 0.9 * 300**-0.6 * 1.3, r"$n^{-0.6}$", color=MUTED)
    ax[0].set_xlabel("$n$"); ax[0].set_ylabel(r"$\|R_n\|$"); ax[0].set_title("(a) exact norm (solid) and bound (dotted)", fontsize=9)
    ax[0].legend(fontsize=7.5)
    ax[1].set_xlabel("$n$"); ax[1].set_title("(b) uniform (solid) vs graded (dashed)", fontsize=9)
    ax[1].legend(fontsize=6, ncol=1, loc="lower left")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_norm.pdf"); plt.close(fig)

    # Fig 2: model integrands
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
    names = {"f1": r"$f_1=e^x\sin x$", "f2": r"$f_2=\cos 4x$", "f4": r"$f_4=\cosh x$"}
    for i, fn in enumerate(["f1", "f2", "f3"]):
        for al in ['0.1', '0.9']:
            d = r1[al][fn]
            ax[i].loglog(d["n"], d["opt"], "o-", c=cols[al], label=fr"optimal, $\alpha={al}$")
            ax[i].loglog(d["n"], d["trap"], "s--", c=cols[al], mfc="white", label=fr"trapezoid, $\alpha={al}$")
        ax[i].set_xlabel("$n$")
    ax[0].set_title(r"(a) $f_1=e^x\sin x$", fontsize=9); ax[1].set_title(r"(b) $f_2=\cos 4x$", fontsize=9)
    ax[2].set_title(r"(c) $f_3=x^{0.7}$", fontsize=9); ax[0].set_ylabel("absolute error")
    _legend(ax[2], [(cols["0.1"], r"$\alpha=0.1$"), (cols["0.9"], r"$\alpha=0.9$")], fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_model.pdf"); plt.close(fig)

    # Fig 3: Abel transform & inversion
    r3, r4 = _load("exp3"), _load("exp4")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for p, c, lab in [("gauss", C1, "Gaussian"), ("hollow", C2, "hollow"), ("cos", C3, r"$\cos^2$")]:
        d = r3[p]
        ax[0].loglog(d["n"], d["opt"], "o-", c=c, label=f"optimal, {lab}")
        ax[0].loglog(d["n"], d["trap"], "s--", c=c, mfc="white", label=f"trapezoid, {lab}")
    d = r3["gauss"]
    ax[0].loglog(d["n"], np.maximum(d["opt_lam"], 1e-30), "^:", c=INK)
    ax[0].set_xlabel("$n$"); ax[0].set_ylabel(r"$\max_m |F(\tau_m)-F_n(\tau_m)|$")
    ax[0].set_title("(a) forward Abel transform", fontsize=9)
    from matplotlib.lines import Line2D
    _legend(ax[0], [(C1, "Gaussian"), (C2, "hollow"), (C3, r"$\cos^2$")], fontsize=6.5, loc="center left", ncol=2)
    ax[0].text(9, 3e-14, r"optimal, $\lambda=\sigma^{-2}$ (Gaussian)", fontsize=6.5, color=INK)
    for p, c, lab in [("gauss", C1, "Gaussian"), ("hollow", C2, "hollow"), ("cos", C3, r"$\cos^2$")]:
        d = r4[p]
        ax[1].loglog(d["n"], d["opt"], "o-", c=c, label=f"optimal, {lab}")
        ax[1].loglog(d["n"], d["trap"], "s--", c=c, mfc="white", label=f"trapezoid, {lab}")
    ax[1].set_xlabel("$n$"); ax[1].set_ylabel(r"$\max_k|\varepsilon_k-\varepsilon(r_k)|$")
    ax[1].set_title("(b) Abel inversion, exact data", fontsize=9)
    _legend(ax[1], [(C1, "Gaussian"), (C2, "hollow"), (C3, r"$\cos^2$")], fontsize=6.5, loc="lower left")
    pr = r4["profile"]
    rr = np.linspace(0, 1, 300)
    ax[2].plot(rr, [(1 + 3 * x**2) * np.exp(-x**2 / .25) for x in rr], c=MUTED, lw=1, label=r"exact $\varepsilon(r)$")
    ax[2].plot(pr["r"], pr["opt"], "o", c=C2, ms=3, label="recovered, $n=64$")
    ax[2].plot(pr["y"], np.array(pr["F"]) / max(pr["F"]), "--", c=C1, lw=1, label=r"data $F(y)/\max F$")
    ax[2].set_xlabel("$r$ (or $y$)"); ax[2].set_title("(c) hollow profile", fontsize=9); ax[2].legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_abel.pdf"); plt.close(fig)

    # Fig 4: fractional relaxation
    r5 = _load("exp5")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    cc = {"0.3": C1, "0.5": C2, "0.8": C3}
    for a in ["0.3", "0.5", "0.8"]:
        for j, mesh in enumerate(["unif", "grad"]):
            d = r5[a][mesh]
            ax[j].loglog(d["n"], d["opt"], "o-", c=cc[a], label=fr"optimal, $\alpha={a}$")
            ax[j].loglog(d["n"], d["trap"], "s--", c=cc[a], mfc="white", label=fr"trapezoid, $\alpha={a}$")
    ax[0].set_title("(a) uniform mesh", fontsize=9); ax[1].set_title(r"(b) graded mesh $t_k=(k/n)^{2/\alpha}$", fontsize=9)
    for j in range(2):
        ax[j].set_xlabel("$n$")
    ax[0].set_ylabel(r"$\max_k |y_k-E_\alpha(-t_k^\alpha)|$")
    _legend(ax[0], [(cc["0.3"], r"$\alpha=0.3$"), (cc["0.5"], r"$\alpha=0.5$"), (cc["0.8"], r"$\alpha=0.8$")], fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_relax.pdf"); plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "fig"]
    for w in which:
        t0 = time.time()
        {"1": exp1, "2": exp2, "3": exp3, "4": exp4, "5": exp5, "6": exp6, "fig": figures}[w]()
        print(f"[{w}] {time.time()-t0:.1f}s", flush=True)
