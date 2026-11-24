# -*- coding: utf-8 -*-
"""Independent re-derivation of the main formulas of the paper.

Nothing from optquad.py is imported.  The optimal weights are obtained in three
ways that do not use the closed form of Theorem 4.2:

  (P) Peano-kernel route.  For f in W_2^{(1,0)} write f = e^{-x}(f(0)+int_0^x e^s v(s) ds),
      v = f'+f.  If R(e^{-x}) = 0 then R(f) = <rho, v>_{L2} with
      rho(s) = e^{s} [ g(s) - sum_{x_j > s} A_j e^{-x_j} ],  g(s) = int_s^t w(x) e^{-x} dx.
      Minimising ||rho|| decouples interval by interval.  All integrals by mp.quad.
  (K) Reproducing-kernel route in H^1(0,t) with the full norm ||f'||^2 + ||f||^2:
      K(x,y) = cosh(min) cosh(t-max) / sinh t, weights = G^{-1} b,  b_i = I(K(., x_i)).
      This is the minimal-norm interpolation algorithm (Remark 5.4(i)).
  (C) the closed form of Theorem 4.2, typed in again from the paper.

It then recomputes a sample of the numbers quoted in the text and in the tables.
Run:  python independent_check.py      (about 3 minutes)
"""
import os, random
import mpmath as mp

mp.mp.dps = 40
ROOT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
LOG = open(os.path.join(ROOT, "results", "independent_check.log"), "w")


def P(*s):
    line = " ".join(str(x) for x in s)
    print(line)
    LOG.write(line + "\n")
    LOG.flush()


def nstr(x, d=6):
    return mp.nstr(x, d)


# ---------------------------------------------------------------- integrals by quadrature
def Iquad(f, t, a, X=None):
    """int_0^t (t-x)^{a-1} f(x) dx in the variable v=(t-x)^a, split at the nodes."""
    pts = [mp.mpf(0), t] if X is None else list(X)
    s = mp.mpf(0)
    for xl, xr in zip(pts, pts[1:]):
        s += mp.quad(lambda v: f(t - v ** (1 / a)), [(t - xr) ** a, (t - xl) ** a])
    return s / a


# ---------------------------------------------------------------- (C) closed form, Theorem 4.2
def E(a, u):
    return u ** a / a * mp.hyp1f1(a, a + 1, u) if u > 0 else mp.mpf(0)


def lg(a, u):
    return mp.gammainc(a, 0, u) if u > 0 else mp.mpf(0)


def weights_closed(X, t, a):
    n = len(X) - 1
    u = [t - x for x in X]
    c = [None]
    for k in range(1, n + 1):
        N = lg(a, u[k - 1]) - lg(a, u[k]) + mp.exp(-2 * u[k]) * E(a, u[k]) - mp.exp(-2 * u[k - 1]) * E(a, u[k - 1])
        D = mp.exp(-2 * u[k]) - mp.exp(-2 * u[k - 1])
        c.append(N / D)
    A = [mp.exp(-t) * E(a, t) - mp.exp(-t) * c[1]]
    A += [mp.exp(-u[k]) * (c[k] - c[k + 1]) for k in range(1, n)]
    A.append(c[n])
    return A


def norm2_closed(X, t, a, terms=80):
    n = len(X) - 1
    u = [t - x for x in X]
    # int_0^t q^2 by the series (eq. q2)
    s = mp.mpf(0)
    for m in range(terms):
        bm = mp.fsum(1 / (mp.factorial(i) * mp.factorial(m - i) * (i + a) * (m - i + a)) for i in range(m + 1))
        s += bm * mp.mpf(2) ** (-(m + 2 * a + 1)) * mp.gammainc(m + 2 * a + 1, 0, 2 * t)
    for k in range(1, n + 1):
        N = lg(a, u[k - 1]) - lg(a, u[k]) + mp.exp(-2 * u[k]) * E(a, u[k]) - mp.exp(-2 * u[k - 1]) * E(a, u[k - 1])
        D = mp.exp(-2 * u[k]) - mp.exp(-2 * u[k - 1])
        s -= N ** 2 / (2 * D)
    return s


# ---------------------------------------------------------------- (P) Peano-kernel route
def make_g(t, a):
    cache = {}

    def g(s):          # g(s) = int_s^t (t-x)^{a-1} e^{-x} dx, by quadrature
        if s >= t:
            return mp.mpf(0)
        key = (mp.mp.dps, s)
        if key not in cache:
            cache[key] = mp.quad(lambda v: mp.exp(-(t - v ** (1 / a))), [0, (t - s) ** a]) / a
        return cache[key]
    return g


def weights_peano(X, t, a):
    n = len(X) - 1
    g = make_g(t, a)
    B = [None]
    nrm2 = mp.mpf(0)
    for k in range(1, n + 1):
        num = mp.quad(lambda s: mp.exp(2 * s) * g(s), [X[k - 1], X[k]])
        den = (mp.exp(2 * X[k]) - mp.exp(2 * X[k - 1])) / 2
        B.append(num / den)
        nrm2 += mp.quad(lambda s: mp.exp(2 * s) * (g(s) - B[k]) ** 2, [X[k - 1], X[k]])
    A = [None] * (n + 1)
    A[n] = B[n] * mp.exp(t)
    for k in range(1, n):
        A[k] = mp.exp(X[k]) * (B[k] - B[k + 1])
    A[0] = Iquad(lambda x: mp.exp(-x), t, a) - mp.fsum(A[k] * mp.exp(-X[k]) for k in range(1, n + 1))
    return A, nrm2


# ---------------------------------------------------------------- (K) RKHS route, H^1 full norm
def weights_rkhs(X, t, a):
    K = lambda x, y: mp.cosh(min(x, y)) * mp.cosh(t - max(x, y)) / mp.sinh(t)
    G = mp.matrix([[K(xi, xj) for xj in X] for xi in X])
    b = mp.matrix([Iquad(lambda x: K(x, xi), t, a, sorted(set([mp.mpf(0), xi, t]))) for xi in X])
    return list(mp.lu_solve(G, b))


def maxdiff(A, B):
    return max(abs(x - y) for x, y in zip(A, B))


# ================================================================ 1. weights
t = mp.mpf(1)
random.seed(2026)
Xr = [mp.mpf(0)] + sorted(mp.mpf(random.random()) for _ in range(7)) + [t]   # n = 8, random
P("== 1. optimal weights by three independent routes (t=1, 8 random intervals)")
for a in ['0.1', '0.4', '0.8']:
    a = mp.mpf(a)
    AC = weights_closed(Xr, t, a)
    AP, n2P = weights_peano(Xr, t, a)
    AK = weights_rkhs(Xr, t, a)
    P("  alpha", a, " |closed-Peano| =", nstr(maxdiff(AC, AP), 3), " |closed-RKHS H1| =", nstr(maxdiff(AC, AK), 3),
      " ||R||^2: closed", nstr(norm2_closed(Xr, t, a), 15), " Peano", nstr(n2P, 15))
    P("     min weight", nstr(min(AC), 4), " sum A_k", nstr(mp.fsum(AC), 10), " < t^a/a =", nstr(t ** a / a, 10))
    for beta in (1, -1):
        err = Iquad(lambda x: mp.exp(beta * x), t, a) - mp.fsum(A * mp.exp(beta * x) for A, x in zip(AC, Xr))
        P("     R(e^{%+dx}) =" % beta, nstr(err, 3))
    err1 = t ** a / a - mp.fsum(AC)
    hmax = max(Xr[k] - Xr[k - 1] for k in range(1, len(Xr)))
    P("     R(1) =", nstr(err1, 6), " bound max(1-sech(h_k/2)) t^a/a =",
      nstr(max(1 - mp.sech((Xr[k] - Xr[k - 1]) / 2) for k in range(1, len(Xr))) * t ** a / a, 6),
      " h^2 t^a/(8a) =", nstr(hmax ** 2 * t ** a / (8 * a), 6))

# perturbation test: the closed-form weights are a local minimiser of ||R||^2
P("== 2. optimality: ||R||^2 grows under perturbations that keep R(e^{-x})=0 (alpha=0.4)")
a = mp.mpf('0.4')
AC = weights_closed(Xr, t, a)


G04 = make_g(t, a)


def norm2_peano_given(A, X):
    n = len(X) - 1
    g = G04
    tot = mp.mpf(0)
    for k in range(1, n + 1):
        Bk = mp.fsum(A[j] * mp.exp(-X[j]) for j in range(k, n + 1))
        tot += mp.quad(lambda s: mp.exp(2 * s) * (g(s) - Bk) ** 2, [X[k - 1], X[k]])
    return tot


mp.mp.dps = 20
base = norm2_peano_given(AC, Xr)
for j in (2, 5, 8):
    for eps in ('1e-3', '-1e-3'):
        A2 = list(AC)
        A2[j] += mp.mpf(eps)
        A2[0] -= mp.mpf(eps) * mp.exp(-Xr[j])          # keep R(e^{-x}) = 0
        P("  perturb A_%d by %s: ||R||^2 - ||R_opt||^2 = %s" % (j, eps, nstr(norm2_peano_given(A2, Xr) - base, 4)))
mp.mp.dps = 40

# ================================================================ 3. constants
P("== 3. constants d_alpha, K_alpha, r*-limits")
for a in ['0.1', '0.5', '0.9']:
    a = mp.mpf(a)
    P("  d_%s = %s" % (a, nstr(-mp.zeta(-1 - a) / (a * (a + 1)), 6)))
# Lemma 6.7: int_0^inf s^{a-1} B2~(s) ds = 2 d_a
for a in ['0.3', '0.7']:
    a = mp.mpf(a)
    B2 = lambda th: th ** 2 - th + mp.mpf(1) / 6
    tot = mp.fsum(mp.quad(lambda s: s ** (a - 1) * B2(s - j), [j, j + 1]) for j in range(0, 500))
    # tail beyond N=500 is about (1-a) N^(a-2)/360, i.e. below 1e-7
    P("  alpha %s: int_0^500 s^(a-1)B2 = %s, 2 d_a = %s" % (a, nstr(tot, 8), nstr(-2 * mp.zeta(-1 - a) / (a * (a + 1)), 8)))


def kappa(m, a):
    return (((m + 1) ** (2 * a + 1) - m ** (2 * a + 1)) / (a ** 2 * (2 * a + 1))
            - (((m + 1) ** (a + 1) - m ** (a + 1)) / (a * (a + 1))) ** 2)


a = mp.mpf('0.1')
# sum to M directly, tail by Euler-Maclaurin with kappa_m ~ m^{2a-2}/12 (1 + ...)
S = {}
for M in (20000, 40000, 80000):
    S[M] = mp.fsum(kappa(m, a) for m in range(M))
r1 = S[80000] + (S[80000] - S[40000]) ** 2 / ((S[40000] - S[20000]) - (S[80000] - S[40000]))
P("  K_0.1 partial sums", nstr(S[20000], 8), nstr(S[40000], 8), nstr(S[80000], 8), " Aitken limit", nstr(r1, 7))
for a in ['0.1', '0.3', '0.5']:
    a = mp.mpf(a)
    P("  limit n||R|| at r*=3/(2a+1), a=%s: %s" % (a, nstr(mp.mpf(3) / 2 * (2 * a + 1) ** (-1.5), 6)))

# ================================================================ 4. norm asymptotics (Remark 6.6(ii))
P("== 4. norm of the error functional, closed form (t=1)")


def mesh_graded(n, r):
    return [t - t * (1 - mp.mpf(k) / n) ** r for k in range(n + 1)]


mp.mp.dps = 30
for n in (256, 1024, 4096):
    h = t / n
    X = [mp.mpf(k) / n for k in range(n + 1)]
    a = mp.mpf('0.1'); nr = norm2_closed(X, t, a)
    P("  n=%d  a=0.1: ||R||^2/h^1.2 = %s ;" % (n, nstr(nr / h ** (2 * a + 1), 6)),
      " (||R||^2 - 0.790134 h^1.2)/h^2 = %s" % nstr((nr - mp.mpf('0.790134') * h ** (2 * a + 1)) / h ** 2, 5))
    a = mp.mpf('0.5'); nr = norm2_closed(X, t, a)
    P("          a=0.5: 12||R||^2/h^2 - ln(1/h) = %s" % nstr(12 * nr / h ** 2 - mp.log(1 / h), 6))
    if n == 4096:
        a = mp.mpf('0.9'); nr = norm2_closed(X, t, a)
        P("          a=0.9: ||R||^2/h^2 = %s   (limit 1/9.6 = %s)" % (nstr(nr / h ** 2, 6), nstr(1 / mp.mpf('9.6'), 6)))
        for a in ['0.1', '0.5']:
            a = mp.mpf(a)
            Xg = mesh_graded(n, 3 / (2 * a + 1))
            P("          graded r*, a=%s: n||R|| = %s" % (a, nstr(n * mp.sqrt(norm2_closed(Xg, t, a)), 6)))

P("  table 5 spot check and graded/uniform factors at n=1024; bound/norm ratios")
n = 1024
X = [mp.mpf(k) / n for k in range(n + 1)]
h = t / n
for a in ['0.1', '0.3', '0.5', '0.9']:
    a = mp.mpf(a)
    nu = mp.sqrt(norm2_closed(X, t, a))
    ng = mp.sqrt(norm2_closed(mesh_graded(n, 3 / (2 * a + 1)), t, a))
    if a > 0.5:
        bd = mp.exp(2 * h) / mp.pi ** 2 * t ** (2 * a - 1) / (2 * a - 1) * h ** 2
    elif a == 0.5:
        bd = mp.exp(2 * h) / mp.pi ** 2 * h ** 2 * mp.log(t / h) + 2 * h ** 2
    else:
        bd = (mp.exp(2 * h) / (mp.pi ** 2 * (1 - 2 * a)) + 1 / (a ** 2 * (2 * a + 1))) * h ** (2 * a + 1)
    P("   a=%s  uniform %s  graded %s  factor %s   bound/norm %s" % (a, nstr(nu, 4), nstr(ng, 4), nstr(nu / ng, 4), nstr(mp.sqrt(bd) / nu, 3)))
a = mp.mpf('0.1')
for r in ('1', '1.5', '2', '2.5'):
    n1, n2 = mp.sqrt(norm2_closed(mesh_graded(256, mp.mpf(r)), t, a)), mp.sqrt(norm2_closed(mesh_graded(1024, mp.mpf(r)), t, a))
    P("   a=0.1 r=%s: EOC(256->1024) = %s" % (r, nstr(mp.log(n1 / n2) / mp.log(4), 3)))
n = 8
X = [mp.mpf(k) / n for k in range(n + 1)]
P("   n=8 uniform: a=0.1 %s, a=0.5 %s, a=0.9 %s ; graded r*: a=0.1 %s, a=0.3 %s" % (
    nstr(mp.sqrt(norm2_closed(X, t, mp.mpf('0.1'))), 3), nstr(mp.sqrt(norm2_closed(X, t, mp.mpf('0.5'))), 3),
    nstr(mp.sqrt(norm2_closed(X, t, mp.mpf('0.9'))), 3),
    nstr(mp.sqrt(norm2_closed(mesh_graded(8, mp.mpf(2.5)), t, mp.mpf('0.1'))), 3),
    nstr(mp.sqrt(norm2_closed(mesh_graded(8, mp.mpf('1.875')), t, mp.mpf('0.3'))), 3)))
mp.mp.dps = 40

# ================================================================ 5. model integrands, tables 2-3
P("== 5. errors for model integrands (uniform mesh, t=1), trapezoid weights by exact moments")


def weights_trap(X, t, a):
    n = len(X) - 1
    u = [t - x for x in X]
    # P^T_k = int_{J_k} u^{a-1}(u_{k-1}-u)/h_k du,  Q^T_k = int_{J_k} u^{a-1}(u-u_k)/h_k du
    def m(p, lo, hi):
        return (hi ** (a + p) - lo ** (a + p)) / (a + p)
    A = [mp.mpf(0)] * (n + 1)
    for k in range(1, n + 1):
        h = u[k - 1] - u[k]
        Pk = (u[k - 1] * m(0, u[k], u[k - 1]) - m(1, u[k], u[k - 1])) / h
        Qk = (m(1, u[k], u[k - 1]) - u[k] * m(0, u[k], u[k - 1])) / h
        A[k] += Pk
        A[k - 1] += Qk
    return A


funcs = {
    'f1': (lambda x: mp.exp(x) * mp.sin(x)),
    'f2': (lambda x: mp.cos(4 * x)),
    'f3': (lambda x: x ** mp.mpf('0.7') if x > 0 else mp.mpf(0)),
    'f4': (lambda x: mp.cosh(x)),
}
exact = {}
for a in ['0.1', '0.5', '0.9']:
    exact[('f1', a)] = Iquad(funcs['f1'], t, mp.mpf(a))
for nm in ('f2', 'f3', 'f4'):
    exact[(nm, '0.5')] = Iquad(funcs[nm], t, mp.mpf('0.5'))
P("  check exact I(f3), a=0.5: quad %s  Beta %s" % (nstr(exact[('f3', '0.5')], 15),
                                                     nstr(mp.beta(mp.mpf('0.5'), mp.mpf('1.7')), 15)))
mp.mp.dps = 30
for (nm, a) in [('f1', '0.1'), ('f1', '0.5'), ('f1', '0.9'), ('f2', '0.5'), ('f3', '0.5'), ('f4', '0.5')]:
    al = mp.mpf(a)
    row = []
    for n in (8, 64, 1024):
        X = [mp.mpf(k) / n for k in range(n + 1)]
        fx = [funcs[nm](x) for x in X]
        eo = abs(exact[(nm, a)] - mp.fsum(w * f for w, f in zip(weights_closed(X, t, al), fx)))
        et = abs(exact[(nm, a)] - mp.fsum(w * f for w, f in zip(weights_trap(X, t, al), fx)))
        row.append("n=%d opt %s trap %s" % (n, nstr(eo, 3), nstr(et, 3)))
    P("  %s a=%s: " % (nm, a) + " | ".join(row))

P("  two-term prediction for f1 at n=1024 (Theorem 6.9): ratio R_opt/R_trap")
f = funcs['f1']
f2d = lambda x: 2 * mp.exp(x) * mp.cos(x)          # f1''
for a in ['0.1', '0.5', '0.9']:
    al = mp.mpf(a)
    d = -mp.zeta(-1 - al) / (al * (al + 1))
    Ig = Iquad(lambda x: f2d(x) - f(x), t, al)
    If2 = Iquad(f2d, t, al)
    n = 1024
    h = t / n
    X = [mp.mpf(k) / n for k in range(n + 1)]
    fx = [f(x) for x in X]
    Ro = exact[('f1', a)] - mp.fsum(w * v for w, v in zip(weights_closed(X, t, al), fx))
    Rt = exact[('f1', a)] - mp.fsum(w * v for w, v in zip(weights_trap(X, t, al), fx))
    po = -h ** 2 / 12 * Ig + d * (f2d(t) - f(t)) * h ** (2 + al)
    pt = -h ** 2 / 12 * If2 + d * f2d(t) * h ** (2 + al)
    P("   a=%s  I(f''-f)/I(f'') = %s  measured %s  two-term %s   R_opt/pred %s  R_trap/pred %s" % (
        a, nstr(Ig / If2, 4), nstr(Ro / Rt, 4), nstr(po / pt, 4), nstr(Ro / po, 7), nstr(Rt / pt, 7)))
mp.mp.dps = 40

# ================================================================ 6. Gauss-Jacobi, table 4
P("== 6. Gauss-Jacobi with weight (t-x)^(a-1), a=0.1, f1, nodes from moments (mp, 60 digits)")
mp.mp.dps = 60
a = mp.mpf('0.1')
ex = Iquad(funcs['f1'], t, a)
for m in (3, 5, 9):
    # monic orthogonal polynomial by moments of xi in [0,1] w.r.t. (1-xi)^(a-1)
    mom = [mp.beta(a, j + 1) for j in range(2 * m + 1)]          # int_0^1 (1-x)^{a-1} x^j dx
    M = mp.matrix([[mom[i + j] for j in range(m)] for i in range(m)])
    rhs = mp.matrix([-mom[i + m] for i in range(m)])
    coef = mp.lu_solve(M, rhs)                                    # p(x) = x^m + sum coef_j x^j
    poly = [mp.mpf(1)] + [coef[j] for j in range(m - 1, -1, -1)]
    nodes = sorted(mp.re(z) for z in mp.polyroots(poly, maxsteps=200, extraprec=200))
    V = mp.matrix([[x ** i for x in nodes] for i in range(m)])
    wts = mp.lu_solve(V, mp.matrix(mom[:m]))
    Q = mp.fsum(wts[i] * funcs['f1'](t * nodes[i]) for i in range(m)) * t ** a
    P("  m=%d  GJ error %s" % (m, nstr(abs(ex - Q), 3)))
mp.mp.dps = 40

# ================================================================ 7. Abel transform spot checks
P("== 7. Abel transform (alpha=1/2), sigma=0.5")
sig = mp.mpf('0.5')
prof = {
    'gauss': lambda r: mp.exp(-r ** 2 / sig ** 2),
    'hollow': lambda r: (1 + 3 * r ** 2) * mp.exp(-r ** 2 / sig ** 2),
    'cos2': lambda r: mp.cos(mp.pi * r / 2) ** 2,
}
a = mp.mpf('0.5')
mp.mp.dps = 30
for nm, eps in prof.items():
    f = lambda s: eps(mp.sqrt(1 - s)) if s < 1 else eps(mp.mpf(0))
    n = 8
    taus = [mp.mpf(m) / n for m in range(n + 1)]
    eo = et = mp.mpf(0)
    for m in range(1, n + 1):
        tau = taus[m]
        X = taus[:m + 1]
        F = Iquad(f, tau, a)
        fx = [f(x) for x in X]
        eo = max(eo, abs(F - mp.fsum(w * v for w, v in zip(weights_closed(X, tau, a), fx))))
        et = max(et, abs(F - mp.fsum(w * v for w, v in zip(weights_trap(X, tau, a), fx))))
    P("  forward %s n=8: opt %s trap %s" % (nm, nstr(eo, 3), nstr(et, 3)))
# inversion, cos^2 profile, n=16
eps = prof['cos2']
f = lambda s: eps(mp.sqrt(1 - s)) if s < 1 else eps(mp.mpf(0))
for kind in ('opt', 'trap'):
    n = 16
    taus = [mp.mpf(m) / n for m in range(n + 1)]
    fk = [f(taus[0])]
    for m in range(1, n + 1):
        X = taus[:m + 1]
        W = weights_closed(X, taus[m], a) if kind == 'opt' else weights_trap(X, taus[m], a)
        F = Iquad(f, taus[m], a)
        fk.append((F - mp.fsum(W[k] * fk[k] for k in range(m))) / W[m])
    P("  inversion cos2 n=16 %s: max error %s" % (kind, nstr(max(abs(fk[k] - f(taus[k])) for k in range(n + 1)), 3)))

# ================================================================ 8. fractional relaxation spot check
P("== 8. fractional relaxation y = E_a(-t^a), kappa=1, T=1")


def ML(al, z):
    return mp.nsum(lambda k: z ** k / mp.gamma(al * k + 1), [0, mp.inf])


for a, r in (('0.8', 2 / mp.mpf('0.8')), ('0.3', 2 / mp.mpf('0.3'))):
    al = mp.mpf(a)
    for meshname, rr in (('uniform', 1), ('graded r=2/a', r)):
        for kind in ('opt', 'trap'):
            n = 16
            T = mp.mpf(1)
            tk = [T * (mp.mpf(k) / n) ** rr for k in range(n + 1)]
            y = [mp.mpf(1)]
            for m in range(1, n + 1):
                X = tk[:m + 1]
                W = weights_closed(X, tk[m], al) if kind == 'opt' else weights_trap(X, tk[m], al)
                s = mp.fsum(W[k] * y[k] for k in range(m))
                y.append((1 - s / mp.gamma(al)) / (1 + W[m] / mp.gamma(al)))
            err = max(abs(y[k] - ML(al, -tk[k] ** al)) for k in range(n + 1))
            if meshname == 'uniform' and kind == 'trap':
                continue
            P("  a=%s %s %s n=16: max error %s" % (a, meshname, kind, nstr(err, 3)))

P("== 9. relaxation threshold y'' = y/2 (via series, a=0.3 and 0.5)")
for a in ('0.3', '0.5', '0.8'):
    al = mp.mpf(a)
    y = lambda s: ML(al, -s ** al)
    F = lambda s: mp.diff(y, s, 2) - y(s) / 2
    xs = [mp.mpf(i) / 100 for i in range(5, 101)]
    vals = [F(s) for s in xs]
    roots = [mp.findroot(F, (xs[i], xs[i + 1]), solver='bisect') for i in range(len(xs) - 1) if vals[i] * vals[i + 1] < 0]
    P("  a=%s  roots of y''-y/2 in [0.05,1]: %s" % (a, [nstr(r_, 4) for r_ in roots]))
P("done")
LOG.close()
