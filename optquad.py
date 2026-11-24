# -*- coding: utf-8 -*-
"""
optquad.py -- Sard-optimal quadrature for Riemann--Liouville fractional
integrals in W_2^{(1,0)}(0,t) (phi-function method), reference implementation.

    I(f) := int_0^t (t-x)^(alpha-1) f(x) dx  ~  Q(f) = sum_k A_k f(x_k)

Two back-ends:
  * mpmath  (arbitrary precision)  : weights_opt, weights_trap, error_norm, ...
  * numpy/scipy (float64, fast)    : weights_opt_fast, weights_trap_fast

All formulas are written in the reversed variable u = t - x, so that no
factor exp(2t) appears and the weights are translation invariant.
"""
import mpmath as mp

# ----------------------------------------------------------------------------
# special functions
# ----------------------------------------------------------------------------
def lgam(a, u):
    """lower incomplete gamma  gamma(a,u) = int_0^u s^(a-1) e^(-s) ds"""
    if u == 0:
        return mp.mpf(0)
    return mp.gammainc(a, 0, u)


def Eplus(a, u):
    """E(u) = int_0^u s^(a-1) e^(s) ds = u^a/a * 1F1(a; a+1; u)"""
    if u == 0:
        return mp.mpf(0)
    return u**a / a * mp.hyp1f1(a, a + 1, u)


def q_fun(a, w):
    """q(w) = e^{-w} E(w) = w^a/a * 1F1(1; a+1; -w)   (Kummer)"""
    if w == 0:
        return mp.mpf(0)
    return w**a / a * mp.hyp1f1(1, a + 1, -w)


# ----------------------------------------------------------------------------
# meshes
# ----------------------------------------------------------------------------
def mesh_uniform(t, n):
    t = mp.mpf(t)
    return [t * k / n for k in range(n + 1)]


def mesh_graded(t, n, r):
    """graded towards the singular end-point x=t:  t - x_k = t (1-k/n)^r"""
    t = mp.mpf(t); r = mp.mpf(r)
    return [t - t * (mp.mpf(n - k) / n)**r for k in range(n + 1)]


# ----------------------------------------------------------------------------
# Optimal weights (closed form, Theorem "closed form")
# ----------------------------------------------------------------------------
def weights_opt(X, t, a, lam=1):
    """
    Optimal weights for arbitrary nodes 0=x_0<...<x_n=t.
    lam > 0 gives the formula exact on span{e^{lam x}, e^{-lam x}}
    (optimal in the space with norm ||f' + lam f||_{L2}); lam=1 is the paper.
    """
    t = mp.mpf(t); a = mp.mpf(a); lam = mp.mpf(lam)
    if lam != 1:                       # scaling (Corollary)
        W = weights_opt([lam * x for x in X], lam * t, a, 1)
        return [lam**(-a) * w for w in W]
    n = len(X) - 1
    U = [t - x for x in X]
    G = [lgam(a, u) for u in U]
    E = [Eplus(a, u) for u in U]
    A = [mp.mpf(0)] * (n + 1)
    for k in range(1, n + 1):
        dG = G[k - 1] - G[k]
        dE = E[k - 1] - E[k]
        sh = mp.sinh(X[k] - X[k - 1])
        P = (mp.exp(U[k - 1]) * dG - mp.exp(-U[k - 1]) * dE) / 2    # int w sinh(x-x_{k-1})
        Q = (mp.exp(-U[k]) * dE - mp.exp(U[k]) * dG) / 2            # int w sinh(x_k-x)
        A[k - 1] += Q / sh
        A[k] += P / sh
    return A


def constants_C(X, t, a):
    """optimal constants C_k of phi_k = p + C_k e^x  (closed form)"""
    t = mp.mpf(t); a = mp.mpf(a)
    U = [t - x for x in X]
    G = [lgam(a, u) for u in U]
    E = [Eplus(a, u) for u in U]
    C = [None]
    for k in range(1, len(X)):
        num = mp.exp(2 * X[k]) * E[k] - mp.exp(2 * X[k - 1]) * E[k - 1] \
              + mp.exp(2 * t) * (G[k - 1] - G[k])
        C.append(mp.exp(-t) * num / (mp.exp(2 * X[k]) - mp.exp(2 * X[k - 1])))
    return C


def weights_from_C(X, t, a):
    """A_0=-phi_1(x_0), A_k=phi_k(x_k)-phi_{k+1}(x_k), A_n=phi_n(x_n)"""
    t = mp.mpf(t); a = mp.mpf(a)
    C = constants_C(X, t, a)
    n = len(X) - 1
    p = lambda x: -mp.exp(x - t) * Eplus(a, t - x)
    A = [-(p(X[0]) + C[1] * mp.exp(X[0]))]
    for k in range(1, n):
        A.append(mp.exp(X[k]) * (C[k] - C[k + 1]))
    A.append(p(X[n]) + C[n] * mp.exp(X[n]))
    return A


# ----------------------------------------------------------------------------
# Theorem 1 of the earlier article (Babaev--Elmurodova, PCAM 2026), literally (double series) -- for verification only
# ----------------------------------------------------------------------------
def H_double_series(u, a, tol=None):
    if u == 0:
        return mp.mpf(0)
    tol = tol or mp.mpf(10)**(-mp.mp.dps - 5)
    s = mp.mpf(0); i = 0
    while True:
        ci = 1 / (mp.factorial(i) * (i + a)); inner = mp.mpf(0); j = 0
        while True:
            term = (-2)**j / mp.factorial(j) * u**(i + j + a + 1) / (i + j + a + 1)
            inner += term
            if j > 2 * u + 5 and abs(term) < tol:
                break
            j += 1
        s += ci * inner
        if i > u + 5 and abs(ci * inner) < tol:
            break
        i += 1
    return s


def weights_paper(X, t, a):
    t = mp.mpf(t); a = mp.mpf(a); n = len(X) - 1
    H = [H_double_series(t - x, a) for x in X]
    G = [None] + [H[k] - H[k - 1] for k in range(1, n + 1)]
    D = [None] + [mp.exp(2 * X[k]) - mp.exp(2 * X[k - 1]) for k in range(1, n + 1)]
    S = lambda x: Eplus(a, t - x)
    A = [mp.exp(X[0] - t) * S(X[0]) + 2 * mp.exp(t + X[0]) / D[1] * G[1]]
    for k in range(1, n):
        A.append(2 * mp.exp(t + X[k]) * (G[k + 1] / D[k + 1] - G[k] / D[k]))
    A.append(-mp.exp(X[n] - t) * S(X[n]) - 2 * mp.exp(t + X[n]) / D[n] * G[n])
    return A


# ----------------------------------------------------------------------------
# Norm of the error functional (exact)
# ----------------------------------------------------------------------------
def int_q2_series(t, a):
    """int_0^t q(w)^2 dw = sum_m c_m 2^{-(m+2a+1)} gamma(m+2a+1, 2t)"""
    t = mp.mpf(t); a = mp.mpf(a)
    s = mp.mpf(0); m = 0
    tol = mp.mpf(10)**(-mp.mp.dps - 3)
    while True:
        cm = mp.fsum(1 / (mp.factorial(i) * mp.factorial(m - i) * (i + a) * (m - i + a))
                     for i in range(m + 1))
        term = cm * 2**(-(m + 2 * a + 1)) * lgam(m + 2 * a + 1, 2 * t)
        s += term
        if m > 4 * t + 10 and abs(term) < tol * abs(s):
            break
        m += 1
    return s


def int_q2_quad(t, a):
    return mp.quad(lambda w: q_fun(a, w)**2, [0, t])


def error_norm(X, t, a):
    """||R|| = ||phi||_{L2}; closed-form sum + int_0^t q^2."""
    t = mp.mpf(t); a = mp.mpf(a)
    U = [t - x for x in X]
    G = [lgam(a, u) for u in U]
    E = [Eplus(a, u) for u in U]
    tot = int_q2_series(t, a)
    for k in range(1, len(X)):
        e1, e0 = mp.exp(-2 * U[k]), mp.exp(-2 * U[k - 1])
        num = G[k - 1] - G[k] + e1 * E[k] - e0 * E[k - 1]
        tot -= num**2 / (2 * (e1 - e0))
    return mp.sqrt(max(tot, mp.mpf(0)))


def error_norm_direct(X, t, a):
    t = mp.mpf(t); a = mp.mpf(a)
    C = constants_C(X, t, a); tot = mp.mpf(0)
    p = lambda x: -mp.exp(x - t) * Eplus(a, t - x)
    for k in range(1, len(X)):
        tot += mp.quad(lambda x: (p(x) + C[k] * mp.exp(x))**2, [X[k - 1], X[k]])
    return mp.sqrt(tot)


def norm_upper_bound(X, t, a):
    """Theorem (upper bound):  Poincare on I_1..I_{n-1} (+ I_n if a>1/2),
       direct bound q(w)<=w^a/a on I_n when a<=1/2."""
    t = mp.mpf(t); a = mp.mpf(a); n = len(X) - 1
    hmax = max(X[k] - X[k - 1] for k in range(1, n + 1))
    s = mp.mpf(0)
    last = n if a > mp.mpf(1) / 2 else n - 1
    for k in range(1, last + 1):
        uL, uR = t - X[k - 1], t - X[k]
        h = X[k] - X[k - 1]
        if uR == 0:
            J = uL**(2 * a - 1) / (2 * a - 1)
        elif a == mp.mpf(1) / 2:
            J = mp.log(uL / uR)
        else:
            J = (uL**(2 * a - 1) - uR**(2 * a - 1)) / (2 * a - 1)
        s += mp.exp(2 * h) * h**2 / mp.pi**2 * J
    if a <= mp.mpf(1) / 2:
        hn = X[n] - X[n - 1]
        s += hn**(2 * a + 1) / (a**2 * (2 * a + 1))
    return mp.sqrt(s)


# ----------------------------------------------------------------------------
# comparison rules
# ----------------------------------------------------------------------------
def weights_trap(X, t, a):
    """product trapezoidal rule (piecewise linear interpolant, exact kernel)"""
    t = mp.mpf(t); a = mp.mpf(a); n = len(X) - 1
    A = [mp.mpf(0)] * (n + 1)
    for k in range(1, n + 1):
        uL, uR = t - X[k - 1], t - X[k]
        m0 = (uL**a - uR**a) / a
        m1 = (uL**(a + 1) - uR**(a + 1)) / (a + 1)
        h = X[k] - X[k - 1]
        A[k - 1] += (m1 - uR * m0) / h
        A[k] += (uL * m0 - m1) / h
    return A


def gauss_jacobi(n, aj, bj):
    aj, bj = mp.mpf(aj), mp.mpf(bj)
    J = mp.matrix(n, n)
    for k in range(n):
        s = 2 * k + aj + bj
        J[k, k] = (bj - aj) / (aj + bj + 2) if k == 0 else (bj**2 - aj**2) / (s * (s + 2))
    for k in range(1, n):
        s = 2 * k + aj + bj
        J[k - 1, k] = J[k, k - 1] = mp.sqrt(4 * k * (k + aj) * (k + bj) * (k + aj + bj) /
                                          (s**2 * (s + 1) * (s - 1)))
    ev, V = mp.eigsy(J)
    mu0 = 2**(aj + bj + 1) * mp.gamma(aj + 1) * mp.gamma(bj + 1) / mp.gamma(aj + bj + 2)
    return [ev[i] for i in range(n)], [mu0 * V[0, i]**2 for i in range(n)]


def quad_gj(f, t, a, m):
    t = mp.mpf(t); a = mp.mpf(a)
    xi, w = gauss_jacobi(m, a - 1, 0)
    return (t / 2)**a * mp.fsum(wi * f(t * (1 + x) / 2) for x, wi in zip(xi, w))


def apply(A, X, f):
    return mp.fsum(Ak * f(x) for Ak, x in zip(A, X))


def I_exp(beta, t, a):
    """int_0^t (t-x)^{a-1} e^{beta x} dx = e^{beta t} beta^{-a} gamma(a, beta t)"""
    t = mp.mpf(t); a = mp.mpf(a)
    if beta == 0:
        return t**a / a
    beta = mp.mpc(beta)
    return mp.exp(beta * t) * beta**(-a) * mp.gammainc(a, 0, beta * t)


# ----------------------------------------------------------------------------
# float64 implementation (numpy / scipy) for large n
# ----------------------------------------------------------------------------
def weights_opt_fast(X, t, a):
    import numpy as np
    from scipy.special import gammainc, gamma, hyp1f1
    X = np.asarray(X, float)
    U = t - X
    U[-1] = 0.0
    G = gammainc(a, U) * gamma(a)
    E = np.where(U > 0, U**a / a * hyp1f1(a, a + 1, U), 0.0)
    dG = G[:-1] - G[1:]
    dE = E[:-1] - E[1:]
    sh = np.sinh(np.diff(X))
    P = (np.exp(U[:-1]) * dG - np.exp(-U[:-1]) * dE) / 2
    Q = (np.exp(-U[1:]) * dE - np.exp(U[1:]) * dG) / 2
    A = np.zeros(len(X))
    A[:-1] += Q / sh
    A[1:] += P / sh
    return A


def weights_trap_fast(X, t, a):
    import numpy as np
    X = np.asarray(X, float)
    U = t - X
    U[-1] = 0.0
    uL, uR = U[:-1], U[1:]
    m0 = (uL**a - uR**a) / a
    m1 = (uL**(a + 1) - uR**(a + 1)) / (a + 1)
    h = np.diff(X)
    A = np.zeros(len(X))
    A[:-1] += (m1 - uR * m0) / h
    A[1:] += (uL * m0 - m1) / h
    return A


# ----------------------------------------------------------------------------
# Numerically stable float64 algorithm (Algorithm 1 in the manuscript):
# P_k, Q_k computed by m-point Gauss rules in the local variable
#   interior intervals : Gauss-Legendre (integrand analytic on the interval)
#   last interval      : Gauss-Jacobi with weight (t-x)^(alpha-1)
# ----------------------------------------------------------------------------
_GL_CACHE = {}
def _rules(a, m):
    from scipy.special import roots_legendre, roots_jacobi
    key = (a, m)
    if key not in _GL_CACHE:
        import numpy as np
        xl, wl = roots_legendre(m)
        # Gauss-Jacobi rule for the weight (1-z)^(a-1): the double-precision eigenvalue
        # computation (scipy.special.roots_jacobi) loses up to three digits when a-1 is
        # close to -1, so the rule is computed once per alpha in 40-digit arithmetic
        # (Golub-Welsch, function gauss_jacobi above) and then rounded to float64.
        with mp.workdps(40):
            xm, wm = gauss_jacobi(m, 0, mp.mpf(a) - 1)          # weight (1+z)^(a-1)
            pairs = sorted((float(-x), float(w)) for x, w in zip(xm, wm))
        xj = np.array([p[0] for p in pairs]); wj = np.array([p[1] for p in pairs])
        _GL_CACHE[key] = (xl, wl, xj, wj)
    return _GL_CACHE[key]


def _gj_moment(a, xj, wj, U, g):
    """int_0^U u^(a-1) g(u) du  by Gauss-Jacobi (u = U(1-z)/2)"""
    import numpy as np
    uu = U[:, None] * (1 - xj[None, :]) / 2
    return (U / 2)**a * np.sum(wj[None, :] * g(uu), axis=1)


def weights_opt_stable(X, t, a, m=16, basis="sinh"):
    """Stable float64 weights.  basis="sinh": optimal formula (sinh-hats);
       basis="lin": product trapezoidal rule (linear hats), same quadrature."""
    import numpy as np
    X = np.asarray(X, float); n = len(X) - 1
    xl, wl, xj, wj = _rules(a, m)
    L, R = X[:-1], X[1:]
    h = R - L
    uL, uR = t - L, t - R
    uR[-1] = 0.0
    fun = np.sinh if basis == "sinh" else (lambda z: z)
    # default: Gauss-Legendre on each interval, in the local variable s = x - x_{k-1}
    # (x - x_{k-1} = s, x_k - x = h - s, t - x = u_{k-1} - s), so that no accuracy is lost
    # on meshes with very small intervals
    s = h[:, None] * (1 + xl[None, :]) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        w = (uL[:, None] - s)**(a - 1) * (h[:, None] / 2) * wl[None, :]
    P = np.sum(w * fun(s), axis=1)
    Q = np.sum(w * fun(h[:, None] - s), axis=1)
    # intervals close to the singular end-point (u_R < 2h, incl. the last one):
    # int_{uR}^{uL} = int_0^{uL} - int_0^{uR}, both by Gauss-Jacobi
    near = np.where(uR < 0.5 * h)[0]
    if len(near):
        UL, UR = uL[near], uR[near]
        # x - L = uL - u,  R - x = u - uR
        P[near] = (_gj_moment(a, xj, wj, UL, lambda u: fun(UL[:, None] - u))
                   - _gj_moment(a, xj, wj, UR, lambda u: fun(UL[:, None] - u)))
        Q[near] = (_gj_moment(a, xj, wj, UL, lambda u: fun(u - UR[:, None]))
                   - _gj_moment(a, xj, wj, UR, lambda u: fun(u - UR[:, None])))
    sh = fun(h)
    A = np.zeros(n + 1)
    A[:-1] += Q / sh
    A[1:] += P / sh
    return A


def weights_trap_stable(X, t, a, m=16):
    return weights_opt_stable(X, t, a, m, basis="lin")
