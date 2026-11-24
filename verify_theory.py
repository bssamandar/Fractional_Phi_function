# -*- coding: utf-8 -*-
"""Numerical verification of every theoretical statement used in the manuscript."""
import os, random, time
import mpmath as mp
from optquad import *

mp.mp.dps = 40
ROOT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
out = open(os.path.join(ROOT, "results", "verify_theory.log"), "w")
def P(*s):
    line = " ".join(str(x) for x in s); print(line); out.write(line + "\n"); out.flush()

t = mp.mpf(1)
random.seed(7)
Xr = [mp.mpf(0)] + sorted(mp.mpf(random.random()) for _ in range(9)) + [t]

P("== 1. three representations of the weights coincide (random nodes, n=10)")
for a in ['0.1', '0.5', '0.9']:
    a = mp.mpf(a)
    A1, A2, A3 = weights_paper(Xr, t, a), weights_opt(Xr, t, a), weights_from_C(Xr, t, a)
    P(" alpha", a, " |paper-closed|", mp.nstr(max(abs(x - y) for x, y in zip(A1, A2)), 3),
      " |C-form - closed|", mp.nstr(max(abs(x - y) for x, y in zip(A3, A2)), 3))

P("== 2. double series collapse  H(u) = (gamma(a,u) - e^{-2u}E(u))/2")
for a in ['0.2', '0.7']:
    for u in ['0.3', '1', '4']:
        a_, u_ = mp.mpf(a), mp.mpf(u)
        P(" a", a, "u", u, mp.nstr(H_double_series(u_, a_) - (lgam(a_, u_) - mp.exp(-2 * u_) * Eplus(a_, u_)) / 2, 3))

P("== 3. exactness: e^{-x}, e^{x}, and exponential splines; not exact for 1, x")
a = mp.mpf('0.3'); A = weights_opt(Xr, t, a)
for name, beta in [("e^-x", -1), ("e^x", 1), ("1", 0)]:
    P(" R(%s) =" % name, mp.nstr(I_exp(beta, t, a).real - apply(A, Xr, lambda x: mp.exp(beta * x)), 3))
# a random exponential spline s = sum c_k s_k : exact because A_k = I(s_k)
def hat(k, x, X):
    if k > 0 and X[k - 1] <= x <= X[k]:
        return mp.sinh(x - X[k - 1]) / mp.sinh(X[k] - X[k - 1])
    if k < len(X) - 1 and X[k] <= x <= X[k + 1]:
        return mp.sinh(X[k + 1] - x) / mp.sinh(X[k + 1] - X[k])
    return mp.mpf(0)
# I(s_k) by adaptive quadrature, interval by interval, in the variable v = (t-x)^a, in which
# (t-x)^(a-1) dx = -dv/a and the integrand is smooth (no closed-form expression is used)
def I_quad(g, X):
    s = mp.mpf(0)
    for xl, xr in zip(X, X[1:]):
        s += mp.quad(lambda v: g(t - v**(1 / a)), [(t - xr)**a, (t - xl)**a]) / a
    return s
for k in range(len(Xr)):
    Ik = I_quad(lambda x: hat(k, x, Xr), Xr)
    P(" A_%d - I(s_%d) =" % (k, k), mp.nstr(A[k] - Ik, 3))

P("== 4. lambda-scaled formula exact for e^{+-lam x}")
lam = mp.mpf(3)
A = weights_opt(Xr, t, a, lam)
for beta in [lam, -lam]:
    P(" lam=3, R(e^{%s x}) =" % mp.nstr(beta, 2), mp.nstr(I_exp(beta, t, a).real - apply(A, Xr, lambda x: mp.exp(beta * x)), 3))

P("== 5. exact norm: series vs quadrature, closed form vs direct")
for a in ['0.1', '0.5', '0.9']:
    a = mp.mpf(a)
    for tt in [1, 3]:
        P(" a", a, "t", tt, " int q^2: series-quad =", mp.nstr(int_q2_series(tt, a) - int_q2_quad(tt, a), 3))
    X = mesh_uniform(t, 6)
    P("   ||R|| closed", mp.nstr(error_norm(X, t, a), 15), " direct", mp.nstr(error_norm_direct(X, t, a), 15))
    P("   random nodes: closed", mp.nstr(error_norm(Xr, t, a), 15), " direct", mp.nstr(error_norm_direct(Xr, t, a), 15))

P("== 6. norm = sup: R(f*) / ||f*|| with f* built from g=-phi equals ||phi||  (uniform n=4, a=0.4)")
a = mp.mpf('0.4'); X = mesh_uniform(t, 4); C = constants_C(X, t, a); A = weights_opt(X, t, a)
pf = lambda x: -mp.exp(x - t) * Eplus(a, t - x)
def phi(x):
    for k in range(1, len(X)):
        if x <= X[k]:
            return pf(x) + C[k] * mp.exp(x)
    return pf(x) + C[-1] * mp.exp(x)
mp.mp.dps = 25
def fstar(x):
    return -mp.exp(-x) * mp.quad(lambda s: mp.exp(s) * phi(s), [0] + [xx for xx in X if xx < x] + [x]) if x > 0 else mp.mpf(0)
Rf = mp.quad(lambda x: (t - x)**(a - 1) * fstar(x), X) - apply(A, X, fstar)
nf = mp.sqrt(mp.quad(lambda x: phi(x)**2, X))
P(" R(f*)/||f*|| =", mp.nstr(Rf / nf, 12), "  ||phi|| =", mp.nstr(error_norm(X, t, a), 12))
mp.mp.dps = 40

P("== 7. rigorous upper bound vs exact norm (uniform, graded, random)")
for a in ['0.1', '0.5', '0.9']:
    a = mp.mpf(a)
    for lab, X in [("unif 64", mesh_uniform(t, 64)), ("grad r=2 64", mesh_graded(t, 64, 2)), ("random 10", Xr)]:
        en, ub = error_norm(X, t, a), norm_upper_bound(X, t, a)
        P(" a", a, lab, " ||R||", mp.nstr(en, 5), " bound", mp.nstr(ub, 5), " ratio", mp.nstr(ub / en, 4))

P("== 8. asymptotics on uniform mesh")
def kappa(m, al):
    """dist^2 of u^al/al from the constants in L2(m, m+1) (closed form)"""
    m = mp.mpf(m)
    I2 = ((m + 1)**(2 * al + 1) - m**(2 * al + 1)) / (al**2 * (2 * al + 1))
    I1 = ((m + 1)**(al + 1) - m**(al + 1)) / (al * (al + 1))
    return I2 - I1**2
def Kalpha(al, M0=500, J=7):
    """K_alpha = sum_{m>=0} kappa_m.  The terms decay only like m^(2 al - 2), so the partial sums
       S(M) = K + sum_i c_i M^(2 al - 1 - i) are extrapolated (Richardson, M doubled J-1 times)."""
    with mp.workdps(50):
        al = mp.mpf(al); S = []; s = mp.mpf(0); m = 0
        for j in range(J):
            while m < M0 * 2**j:
                s += kappa(m, al); m += 1
            S.append(s)
        for i in range(J - 1):
            p = 2 * al - 1 - i
            S = [(S[k + 1] - 2**p * S[k]) / (1 - 2**p) for k in range(len(S) - 1)]
        return +S[-1]
mp.mp.dps = 25
for al in ['0.1', '0.3']:
    al = mp.mpf(al); K = Kalpha(al)
    for n in [256, 1024, 4096]:
        e = error_norm(mesh_uniform(t, n), t, al)
        P(" a", al, "n", n, " ||R||^2/h^(2a+1) =", mp.nstr(e**2 * mp.mpf(n)**(2 * al + 1), 6), " K_a =", mp.nstr(K, 6),
          " (||R||^2 - K_a h^(2a+1))/h^2 =", mp.nstr((e**2 - K * mp.mpf(n)**(-(2 * al + 1))) * n**2, 6),
          " -1/(12(1-2a)) =", mp.nstr(-1 / (12 * (1 - 2 * al)), 6))
for al in ['0.7', '0.9']:
    al = mp.mpf(al)
    for n in [256, 1024, 4096]:
        e = error_norm(mesh_uniform(t, n), t, al)
        P(" a", al, "n", n, " ||R||^2/h^2 =", mp.nstr(e**2 * n**2, 6), " t^(2a-1)/(12(2a-1)) =", mp.nstr(1 / (12 * (2 * al - 1)), 6))
al = mp.mpf('0.5')
for n in [256, 1024, 4096]:
    e = error_norm(mesh_uniform(t, n), t, al); h = mp.mpf(1) / n
    P(" a 0.5 n", n, " 12||R||^2/h^2 - ln(1/h) =", mp.nstr(12 * e**2 / h**2 - mp.log(1 / h), 6))
mp.mp.dps = 40

P("== 9. graded meshes: rate of ||R|| (a=0.1, r* = 2/(2a+1) = 1.667)")
a = mp.mpf('0.1')
for r in ['1', '1.5', '2', '2.5']:
    es = []
    for n in [64, 256, 1024]:
        es.append(error_norm(mesh_graded(t, n, r), t, a))
    P(" r", r, " ||R||:", [mp.nstr(e, 4) for e in es], " rate:", mp.nstr(mp.log(es[1] / es[2]) / mp.log(4), 3))

P("== 10. smooth f: bound |R| <= (1-sech(h/2)) ||f''-f|| t^a/a  and  asymptotic  R ~ -(h^2/12) I(f''-f)")
b = 1 + 1j
tests = {
    "sin(x)e^x": (lambda x: mp.sin(x) * mp.exp(x), lambda a: mp.im(I_exp(b, t, a)),
                  lambda a: 2 * mp.re(I_exp(b, t, a)) - mp.im(I_exp(b, t, a)), lambda a: 2 * mp.re(I_exp(b, t, a)),
                  lambda x: mp.exp(x) * (2 * mp.cos(x) - mp.sin(x))),
    "cos(4x)": (lambda x: mp.cos(4 * x), lambda a: mp.re(I_exp(4j, t, a)), lambda a: -17 * mp.re(I_exp(4j, t, a)),
                lambda a: -16 * mp.re(I_exp(4j, t, a)), lambda x: -17 * mp.cos(4 * x)),
}
for a in ['0.1', '0.5', '0.9']:
    a = mp.mpf(a)
    for name, (f, Iex, ILf, If2, Lf) in tests.items():
        sup = max(abs(Lf(mp.mpf(i) / 2000)) for i in range(2001))
        for n in [64, 1024, 8192]:
            X = mesh_uniform(t, n); h = mp.mpf(1) / n
            Ro = Iex(a) - apply(weights_opt(X, t, a), X, f)
            Rt = Iex(a) - apply(weights_trap(X, t, a), X, f)
            P(" a", a, name, "n", n, " R_opt/pred", mp.nstr(Ro / (-(h**2 / 12) * ILf(a)), 6),
              " R_trap/pred", mp.nstr(Rt / (-(h**2 / 12) * If2(a)), 6),
              " |R_opt|/bound", mp.nstr(abs(Ro) / ((1 - mp.sech(h / 2)) * sup * t**a / a), 3))
P("== 11. graded meshes: n ||R|| -> sqrt(r^3 t^(2a+1) / (12 (r(2a+1)-2)));  r = 3/(2a+1) is optimal")
mp.mp.dps = 30
for a in ['0.1', '0.3', '0.5', '0.7', '0.9']:
    a = mp.mpf(a)
    for r in [3 / (2 * a + 1), mp.mpf(3)]:
        lim = mp.sqrt(r**3 * t**(2 * a + 1) / (12 * (r * (2 * a + 1) - 2)))
        vals = [n * error_norm(mesh_graded(t, n, r), t, a) for n in [256, 1024, 4096]]
        P(" a", a, " r", mp.nstr(r, 5), " n||R||:", [mp.nstr(v, 7) for v in vals], " limit", mp.nstr(lim, 7))

P("== 12. two-term expansion  R = -(h^2/12) I(g) + d_a g(t) h^(2+a) + O(h^3),  d_a = -zeta(-1-a)/(a(a+1))")
f1, f1pp = (lambda x: mp.sin(x) * mp.exp(x)), (lambda x: 2 * mp.exp(x) * mp.cos(x))
for a in ['0.1', '0.5', '0.9']:
    a = mp.mpf(a); d = -mp.zeta(-1 - a) / (a * (a + 1))
    If, Ifpp = mp.im(I_exp(b, t, a)), 2 * mp.re(I_exp(b, t, a))
    P(" a", a, " d_a =", mp.nstr(d, 8), "  I(f''-f)/I(f'') =", mp.nstr((Ifpp - If) / Ifpp, 5))
    for n in [64, 256, 1024, 4096]:
        X = mesh_uniform(t, n); h = mp.mpf(1) / n
        Ro = If - apply(weights_opt(X, t, a), X, f1)
        Rt = If - apply(weights_trap(X, t, a), X, f1)
        po = -(h**2 / 12) * (Ifpp - If) + d * (f1pp(t) - f1(t)) * h**(2 + a)
        pt = -(h**2 / 12) * Ifpp + d * f1pp(t) * h**(2 + a)
        P("   n", n, " R_opt/two-term", mp.nstr(Ro / po, 8), " R_trap/two-term", mp.nstr(Rt / pt, 8),
          " (R_opt - two-term)/h^3", mp.nstr((Ro - po) / h**3, 4), " R_opt/R_trap", mp.nstr(Ro / Rt, 4),
          " predicted", mp.nstr(po / pt, 4))
P("== 13. relaxation: where does y = E_a(-t^a) satisfy y'' < y/2  (i.e. |y''-y| > |y''|)?")
mp.mp.dps = 30
def ml(a, z):
    return mp.nsum(lambda k: z**k / mp.gamma(a * k + 1), [0, mp.inf])
for a in ['0.3', '0.5', '0.8']:
    a = mp.mpf(a)
    F = lambda s: mp.diff(lambda x: ml(a, -x**a), s, 2) - ml(a, -s**a) / 2
    grid = [mp.mpf(i) / 50 for i in range(5, 51)]
    signs = [F(s) for s in grid]
    root = None
    for s0, s1, v0, v1 in zip(grid, grid[1:], signs, signs[1:]):
        if v0 > 0 and v1 < 0:
            root = mp.findroot(F, (s0, s1), solver='anderson')
            break
    P(" a", a, " y''-y/2 at t=1:", mp.nstr(signs[-1], 4), " sign change at t =", mp.nstr(root, 4) if root else "none in [0.1,1]")
P("done")
