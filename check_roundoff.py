# Double-precision behaviour of the representations of the weights (remark on round-off):
# double series, formula with the constants C_k, formula (P_k, Q_k), Algorithm 1.
import os, numpy as np, mpmath as mp, math
from scipy.special import gammainc, gamma, hyp1f1
from optquad import weights_opt, weights_opt_fast, weights_opt_stable
mp.mp.dps = 40
ROOT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
LOG = open(os.path.join(ROOT, 'results', 'check_roundoff.log'), 'w')
def Hser64(u, a):
    if u == 0: return 0.0
    s = 0.0
    for i in range(0, 400):
        ci = 1.0/(math.gamma(i+1)*(i+a)) if i < 170 else 0.0
        inner = 0.0
        for j in range(0, 600):
            if j >= 170: break
            term = (-2.0)**j/math.gamma(j+1)*u**(i+j+a+1)/(i+j+a+1)
            inner += term
            if j > 2*u+5 and abs(term) < 1e-18*max(1.0, abs(inner)): break
        s += ci*inner
        if i > u+5 and abs(ci*inner) < 1e-18*max(1.0, abs(s)): break
    return s
def E64(a, u): return u**a/a*hyp1f1(a, a+1, u) if u > 0 else 0.0
def w_series64(X, t, a):
    n = len(X)-1
    H = [Hser64(t-x, a) for x in X]
    A = np.zeros(n+1)
    D = [None]+[math.exp(2*X[k])-math.exp(2*X[k-1]) for k in range(1, n+1)]
    G = [None]+[H[k]-H[k-1] for k in range(1, n+1)]
    A[0] = math.exp(X[0]-t)*E64(a, t-X[0]) + 2*math.exp(t+X[0])/D[1]*G[1]
    for k in range(1, n): A[k] = 2*math.exp(t+X[k])*(G[k+1]/D[k+1]-G[k]/D[k])
    A[n] = -2*math.exp(t+X[n])/D[n]*G[n]
    return A
def w_C64(X, t, a):
    X = np.asarray(X, float); n = len(X)-1; U = t-X; U[-1] = 0
    G = gammainc(a, U)*gamma(a); E = np.array([E64(a, u) for u in U])
    C = np.zeros(n+1)
    for k in range(1, n+1):
        num = math.exp(2*X[k])*E[k]-math.exp(2*X[k-1])*E[k-1]+math.exp(2*t)*(G[k-1]-G[k])
        C[k] = math.exp(-t)*num/(math.exp(2*X[k])-math.exp(2*X[k-1]))
    A = np.zeros(n+1); A[0] = math.exp(-t)*E[0]-C[1]
    for k in range(1, n): A[k] = math.exp(X[k])*(C[k]-C[k+1])
    A[n] = C[n]*math.exp(t); return A
def rel(A, ref):
    ref = np.array([float(r) for r in ref]); return float(np.max(np.abs(A-ref)/np.abs(ref)))
for (t, n, a) in [(1.0, 10, 0.1), (1.0, 100, 0.1), (1.0, 1000, 0.1), (5.0, 10, 0.1), (10.0, 10, 0.1), (10.0, 100, 0.1), (10.0, 10, 0.5), (20.0, 10, 0.1), (20.0, 20, 0.5)]:
    X = np.linspace(0, t, n+1)
    ref = weights_opt([mp.mpf(x) for x in X], t, a)          # reference for the same double nodes
    out = [f"t={t:g} n={n} a={a}"]
    if n <= 100: out.append("series %.1e" % rel(w_series64(list(X), t, a), ref))
    out.append("C-form %.1e" % rel(w_C64(X, t, a), ref))
    out.append("PQ-form %.1e" % rel(weights_opt_fast(X, t, a), ref))
    out.append("Alg1 %.1e" % rel(weights_opt_stable(X, t, a), ref))
    line = "  ".join(out)
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()

# accuracy of the double-precision Gauss-Jacobi rule of SciPy for the weight (1-z)^(alpha-1),
# compared with the rule computed in 40-digit arithmetic (Golub-Welsch)
from scipy.special import roots_jacobi
from optquad import gauss_jacobi
for a in [0.1, 0.5]:
    xs, ws = roots_jacobi(16, a - 1.0, 0.0)
    with mp.workdps(40):
        xm, wm = gauss_jacobi(16, 0, mp.mpf(a) - 1)      # weight (1+z)^(a-1); mirror z -> -z
        pairs = sorted((-x, w) for x, w in zip(xm, wm))
    ex = max(abs(float((mp.mpf(x) - X) / (1 - X))) for x, (X, _) in zip(xs, pairs))
    ew = max(abs(float((mp.mpf(w) - W) / W)) for w, (_, W) in zip(ws, pairs))
    line = "scipy roots_jacobi(16, %.1f-1, 0): max rel. err. of 1-z_i %.1e, of weights %.1e" % (a, ex, ew)
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()
