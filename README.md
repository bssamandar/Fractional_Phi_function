# Optimal quadrature for Riemann–Liouville fractional integrals on arbitrary nodes

Code and raw results for the manuscript

> S. S. Babaev, S. S. Elmurodova, *Optimal quadrature for Riemann–Liouville fractional integrals on
> arbitrary nodes: error analysis and applications to Abel integral equations* (submitted).

The quadrature formula

$$\int_0^t (t-x)^{\alpha-1} f(x)\,dx \approx \sum_{k=0}^n A_k f(x_k),\qquad 0=x_0<\dots<x_n=t,\ 0<\alpha<1,$$

is optimal in the sense of Sard in $W_2^{(1,0)}(0,t)$ with the semi-norm $\|f'+f\|_{L_2}$
($\varphi$-function method). The scripts compute its weights, the norm of its error functional, and
all numerical experiments of the paper.

## Quick start

```bash
pip install -r requirements.txt
python run_all.py            # all tables, figures, CSV files and checks (about 6 minutes on one core)
python run_all.py --quick    # only the verification of the theory and the round-off check
```

Results were produced with Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1, mpmath 1.4.1 and
Matplotlib 3.11.2 (`requirements.txt`).

## Files

| file | content |
|---|---|
| `optquad.py` | library: optimal weights (closed form, arbitrary precision; Algorithm 1 in double precision), product trapezoidal weights, norm of the error functional, Gauss–Jacobi rule |
| `experiments.py` | the six numerical experiments and the four figures (`python experiments.py 1 2 3 4 5 6 fig`) |
| `make_tables.py` | rows of the LaTeX tables, written to `tables/` |
| `export_csv.py` | raw results as CSV files, written to `results/csv/` |
| `verify_theory.py` | numerical verification of the theoretical statements, log in `results/verify_theory.log` |
| `check_roundoff.py` | accuracy of the three representations of the weights in double precision (Remark 4.2), log in `results/check_roundoff.log` |
| `run_all.py` | runs everything in the right order |

## How each table and figure is produced

| in the paper | experiment | raw data | table rows / figure |
|---|---|---|---|
| Table 1 (cost and round-off of Algorithm 1) | `experiments.py 6` | `results/csv/exp6_cost_roundoff.csv` | `tables/tab_cost.tex` |
| Tables 2, 3, Figure 1 (model integrands) | `experiments.py 1` | `results/csv/exp1_model_integrands.csv` | `tables/tab_f1.tex`, `tables/tab_f234.tex`, `figures/fig_model.pdf` |
| Table 4 (Gauss–Jacobi comparison) | `experiments.py 1` | `results/csv/exp1_gauss_jacobi.csv` | `tables/tab_gj.tex` |
| Table 5, Figure 2 (norm of the error functional) | `experiments.py 2` | `results/csv/exp2_error_norm.csv` | `tables/tab_norm.tex`, `figures/fig_norm.pdf` |
| Table 6, Figure 3(a) (forward Abel transform) | `experiments.py 3` | `results/csv/exp3_abel_forward.csv` | `tables/tab_abel_fwd.tex`, `figures/fig_abel.pdf` |
| Table 7, Figure 3(b,c) (Abel inversion) | `experiments.py 4` | `results/csv/exp4_abel_inversion.csv`, `results/csv/exp4_hollow_profile_n64.csv` | `tables/tab_abel_inv.tex`, `figures/fig_abel.pdf` |
| Table 8, Figure 4 (fractional relaxation) | `experiments.py 5` | `results/csv/exp5_relaxation.csv` | `tables/tab_relax.tex`, `figures/fig_relax.pdf` |
| Section 8.1 and the numbers quoted in Sections 4–6 | `verify_theory.py`, `check_roundoff.py` | `results/verify_theory.log`, `results/check_roundoff.log` | – |

## Independence of the verification

`verify_theory.py` compares every closed-form expression with a quantity computed without it:
the weights with the double series of the earlier construction and with the integrals $I(s_k)$ of
the exponential hat functions (adaptive quadrature in the variable $(t-x)^\alpha$); the norm of the
error functional with the numerically integrated $\|\varphi\|_{L_2}$; the extremal function with
$I(f_*)$ computed by quadrature; the error expansions with exact values of the integrals.
Reference values use 40–60 significant digits (mpmath).

## Contact

Samandar S. Babaev, Tashkent International University — bssamandar@gmail.com
