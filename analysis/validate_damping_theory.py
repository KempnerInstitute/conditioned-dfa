"""Historical finite-sample damping simulation for conditioned DFA.

This script explores a heuristic damping relation considered during development.
The current paper does not state that relation as a proposition or empirical law;
it retains damping as a validation-selected parameter. In the linear-Gaussian
model the population nDFA input factor is

    M(lambda)      = Sigma (Sigma + lambda I)^{-1},

but in practice the inverse is a plug-in built from an empirical uncentered second
moment Sigma_hat = (1/n) sum_i x_i x_i^T of n i.i.d. N(0, Sigma) samples in
dimension d, so the *realized* factor is

    M_hat(lambda)  = Sigma (Sigma_hat + lambda I)^{-1}.

The population factor is monotonically best-conditioned as lambda -> 0 (perfect
spectrum flattening), while the plug-in inverse amplifies estimation noise in the
low-eigenvalue subspace, worst as lambda -> 0. Balancing these two effects gives
an interior optimum lambda*(n, spectrum). The derivation (see damping_theory
snippet) predicts, in the n >= d regime where ||Sigma_hat - Sigma|| ~
||Sigma|| sqrt(d/n),

    lambda*  ~  sqrt(lambda_min lambda_max) (d/n)^{1/4}
            =  lambda_min sqrt(kappa) (d/n)^{1/4},

i.e. a log-log slope of +1/4 in (d/n) and -1/2 in kappa (at fixed lambda_max=1).

We test this two ways:
  (A) realized condition number kappa(M_hat) averaged over draws (clean, fast);
  (B) steps-to-target-loss of the plug-in nDFA GD iteration (the original
      theory-figure metric), refreshing Sigma_hat every step as the rule does.

Outputs (results/infodfa_damping_theory_v1/):
  - damping_theory_curves.csv        risk/steps vs lambda for every (d,n,kappa)
  - damping_theory_optima.csv        extracted lambda* and fitted rates
  - damping_theory_validation.pdf/png validation figure

CPU only, pure numpy. Runtime ~1-2 min.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Spectrum + sampling helpers
# --------------------------------------------------------------------------- #
def make_spectrum(d: int, kappa: float) -> np.ndarray:
    """Log-uniform eigenvalues in [1/kappa, 1], so lambda_max=1, lambda_min=1/kappa."""
    return np.geomspace(1.0, 1.0 / kappa, d)


def sample_sighat_eig(ev: np.ndarray, n: int, rng: np.random.Generator):
    """Draw one empirical uncentered second moment and return its eigendecomp.

    x_i ~ N(0, diag(ev)); Sigma is diagonal in this basis so Sigma stays diag(ev)
    and only Sigma_hat is random. Returns (w, V) with Sigma_hat = V diag(w) V^T.
    """
    d = ev.shape[0]
    X = rng.standard_normal((n, d)) * np.sqrt(ev)
    S = (X.T @ X) / n
    w, V = np.linalg.eigh(S)
    return np.maximum(w, 0.0), V


def realized_factor_cond(ev, w, V, lam):
    """Condition number (max/min eigval of symmetric part) of M_hat = Sigma R_hat.

    Sigma = diag(ev); R_hat = (Sigma_hat + lam I)^{-1} = V diag(1/(w+lam)) V^T.
    We report kappa of the symmetric part (M_hat + M_hat^T)/2, whose eigenvalue
    spread governs the GD convergence rate of eps <- eps (I - eta M_hat).
    """
    Rhat = (V / (w + lam)) @ V.T
    M = np.diag(ev) @ Rhat
    Msym = 0.5 * (M + M.T)
    e = np.linalg.eigvalsh(Msym)
    e = e[e > 1e-12]
    if e.size == 0:
        return np.inf
    return float(e.max() / e.min())


# --------------------------------------------------------------------------- #
# Metric A: expected realized condition number vs lambda
# --------------------------------------------------------------------------- #
def cond_curve(d, n, kappa, lams, draws=64, seed=0):
    rng = np.random.default_rng(seed)
    ev = make_spectrum(d, kappa)
    out = np.zeros(len(lams))
    enorm = 0.0
    for s in range(draws):
        w, V = sample_sighat_eig(ev, n, rng)
        # ||E|| for the sqrt(d/n) chain check (E = Sigma_hat - Sigma, both in eig basis of Sigma=diag)
        Shat = (V * w) @ V.T
        enorm += np.linalg.norm(Shat - np.diag(ev), 2)
        for j, lam in enumerate(lams):
            out[j] += realized_factor_cond(ev, w, V, lam)
    return out / draws, enorm / draws


# --------------------------------------------------------------------------- #
# Metric B: steps-to-target for the plug-in nDFA GD iteration
# --------------------------------------------------------------------------- #
def steps_to_target(d, n, kappa, lam, etas, thresh=1e-3, max_steps=3000,
                    draws=24, seed=0, refresh=True):
    """Median steps for eps <- eps (I - eta Sigma (Sigma_hat+lam I)^{-1}) to reach
    loss L=0.5 eps Sigma eps^T below thresh*L0. Sigma_hat refreshed each step
    (refresh=True) mimics the batch rule. Best eta per draw, median over draws."""
    ev = make_spectrum(d, kappa)
    rng = np.random.default_rng(seed)
    best_per_draw = []
    for s in range(draws):
        # fixed plug-in pool for this draw (fresh each step drawn on the fly)
        best = max_steps + 1
        for eta in etas:
            eps = np.ones(d) / np.sqrt(d)
            L0 = 0.5 * float(eps**2 @ ev)
            Wfix = None
            if not refresh:
                w, V = sample_sighat_eig(ev, n, rng)
                Wfix = (w, V)
            hit = max_steps + 1
            for t in range(max_steps):
                if refresh:
                    w, V = sample_sighat_eig(ev, n, rng)
                else:
                    w, V = Wfix
                # eps <- eps - eta * eps @ M,  M = Sigma (Sigma_hat+lam)^{-1}
                y = eps * ev              # eps @ Sigma
                y = (y @ V) / (w + lam)
                y = y @ V.T               # eps @ Sigma @ Rhat
                eps = eps - eta * y
                L = 0.5 * float(eps**2 @ ev)
                if not np.isfinite(L) or L > 1e4 * L0:
                    break
                if L < thresh * L0:
                    hit = t + 1
                    break
            best = min(best, hit)
        best_per_draw.append(best)
    return float(np.median(best_per_draw))


# --------------------------------------------------------------------------- #
# lambda* extraction (log-parabola fit around the grid argmin)
# --------------------------------------------------------------------------- #
def extract_optimum(lams, risk):
    risk = np.asarray(risk, float)
    lams = np.asarray(lams, float)
    ok = np.isfinite(risk)
    lams, risk = lams[ok], risk[ok]
    i = int(np.argmin(risk))
    if 0 < i < len(lams) - 1:
        x = np.log(lams[i - 1:i + 2])
        y = risk[i - 1:i + 2]
        a, b, _ = np.polyfit(x, y, 2)
        if a > 0:
            xstar = -b / (2 * a)
            xstar = np.clip(xstar, x[0], x[-1])
            return float(np.exp(xstar)), i
    return float(lams[i]), i


def fit_loglog(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
    if m.sum() < 2:
        return np.nan, np.nan
    slope, intercept = np.polyfit(np.log(xs[m]), np.log(ys[m]), 1)
    return float(slope), float(intercept)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--draws", type=int, default=64)
    ap.add_argument("--step-draws", type=int, default=16)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    outdir = Path(args.outdir) if args.outdir else repo / "results" / "infodfa_damping_theory_v1"
    outdir.mkdir(parents=True, exist_ok=True)

    ds = [32, 128]
    kappas = [10.0, 50.0, 100.0]
    nod_grid = [2, 4, 8, 16, 32, 64]        # n/d, kept >= 2 (n >= d concentration regime)
    lams = np.geomspace(1e-4, 3.0, 28)
    if args.quick:
        ds = [32]
        kappas = [10.0, 100.0]
        nod_grid = [2, 8, 32]
        lams = np.geomspace(1e-4, 3.0, 18)
        args.draws, args.step_draws = 24, 8

    curve_rows, opt_rows = [], []
    for d in ds:
        for kappa in kappas:
            for nod in nod_grid:
                n = int(round(d * nod))
                condc, enorm = cond_curve(d, n, kappa, lams, draws=args.draws,
                                          seed=hash((d, kappa, nod)) % (2**31))
                lam_star_cond, _ = extract_optimum(lams, condc)
                for lam, cv in zip(lams, condc):
                    curve_rows.append(dict(d=d, kappa=kappa, n_over_d=nod, n=n,
                                           lam=lam, metric="cond", value=cv))
                # steps metric on a modest lambda subgrid around the cond optimum
                sub = np.geomspace(max(lam_star_cond / 30, lams[0]),
                                   min(lam_star_cond * 30, lams[-1]), 11)
                etas = np.geomspace(0.05, 3.0, 10)
                steps = [steps_to_target(d, n, kappa, lam, etas,
                                         draws=args.step_draws,
                                         seed=(hash((d, kappa, nod, i)) % (2**31)))
                         for i, lam in enumerate(sub)]
                lam_star_steps, _ = extract_optimum(sub, steps)
                for lam, sv in zip(sub, steps):
                    curve_rows.append(dict(d=d, kappa=kappa, n_over_d=nod, n=n,
                                           lam=lam, metric="steps", value=sv))
                opt_rows.append(dict(
                    d=d, kappa=kappa, n_over_d=nod, n=n,
                    d_over_n=d / n, lam_min=1.0 / kappa, lam_max=1.0,
                    E_norm=enorm, E_pred=np.sqrt(d / n) + d / n,
                    lam_star_cond=lam_star_cond, lam_star_steps=lam_star_steps,
                    lam_star_pred=np.sqrt((1.0 / kappa) * 1.0) * (d / n) ** 0.25,
                ))
                print(f"d={d:4d} kappa={kappa:6.0f} n/d={nod:3d} "
                      f"||E||={enorm:.3f} (pred~{np.sqrt(d/n):.3f}) "
                      f"lam*_cond={lam_star_cond:.3e} lam*_steps={lam_star_steps:.3e}",
                      flush=True)

    curves = pd.DataFrame(curve_rows)
    opt = pd.DataFrame(opt_rows)
    curves.to_csv(outdir / "damping_theory_curves.csv", index=False)

    # ------- fit the rates -------
    fit_rows = []
    for d in ds:
        for kappa in kappas:
            sub = opt[(opt.d == d) & (opt.kappa == kappa)].sort_values("d_over_n")
            s_c, _ = fit_loglog(sub.d_over_n, sub.lam_star_cond)
            s_s, _ = fit_loglog(sub.d_over_n, sub.lam_star_steps)
            fit_rows.append(dict(d=d, kappa=kappa, fit="lam_star_vs_d_over_n",
                                 slope_cond=s_c, slope_steps=s_s, predicted=0.25))
    # kappa dependence at fixed d, n/d (pool the mid n/d rows)
    for d in ds:
        for nod in nod_grid:
            sub = opt[(opt.d == d) & (opt.n_over_d == nod)].sort_values("kappa")
            s_c, _ = fit_loglog(sub.kappa, sub.lam_star_cond)
            s_s, _ = fit_loglog(sub.kappa, sub.lam_star_steps)
            fit_rows.append(dict(d=d, kappa=np.nan, n_over_d=nod,
                                 fit="lam_star_vs_kappa",
                                 slope_cond=s_c, slope_steps=s_s, predicted=-0.5))
    # ||E|| vs d/n slope (should be ~0.5 in the sqrt regime)
    s_E, _ = fit_loglog(opt.d_over_n, opt.E_norm)
    fit_rows.append(dict(fit="E_norm_vs_d_over_n", slope_cond=s_E, predicted=0.5))

    fits = pd.DataFrame(fit_rows)
    opt.to_csv(outdir / "damping_theory_optima.csv", index=False)
    fits.to_csv(outdir / "damping_theory_fits.csv", index=False)

    # ------- figure -------
    make_figure(curves, opt, fits, ds, kappas, nod_grid, outdir)

    print("\n=== fitted rates ===")
    with pd.option_context("display.width", 200):
        print(fits.to_string(index=False))
    print("\nWrote:", outdir)


def make_figure(curves, opt, fits, ds, kappas, nod_grid, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    axA, axB, axC = axes

    # A: realized condition-number risk vs lambda, showing the interior optimum
    d0, k0 = ds[0], kappas[-1]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(nod_grid)))
    for c, nod in zip(cmap, nod_grid):
        sub = curves[(curves.d == d0) & (curves.kappa == k0)
                     & (curves.n_over_d == nod) & (curves.metric == "cond")]
        axA.plot(sub.lam, sub.value, "-o", ms=3, color=c, label=f"n/d={nod}")
        so = opt[(opt.d == d0) & (opt.kappa == k0) & (opt.n_over_d == nod)]
        axA.axvline(so.lam_star_cond.iloc[0], color=c, ls=":", lw=0.8)
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlabel(r"damping $\lambda$")
    axA.set_ylabel(r"realized $\kappa(\hat M)$")
    axA.set_title(f"Interior optimum (d={d0}, $\\kappa$={k0:.0f})", fontsize=9)
    axA.legend(fontsize=6, ncol=2)
    axA.grid(alpha=0.3)

    # B: lambda* vs d/n, log-log, with predicted 1/4 slope
    for d in ds:
        for kappa in kappas:
            sub = opt[(opt.d == d) & (opt.kappa == kappa)].sort_values("d_over_n")
            axB.plot(sub.d_over_n, sub.lam_star_cond, "o-", ms=3,
                     label=f"d={d},$\\kappa$={kappa:.0f}")
    x = np.geomspace(opt.d_over_n.min(), opt.d_over_n.max(), 20)
    axB.plot(x, 0.15 * x**0.25, "k--", lw=1.2, label=r"slope 1/4 (pred)")
    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_xlabel(r"$d/n$")
    axB.set_ylabel(r"$\lambda^\star$ (cond metric)")
    axB.set_title("Rate in d/n", fontsize=9)
    axB.legend(fontsize=5.5, ncol=2)
    axB.grid(alpha=0.3)

    # C: fitted slopes vs predicted
    fs = fits[fits.fit == "lam_star_vs_d_over_n"]
    labels = [f"d{r.d},k{r.kappa:.0f}" for r in fs.itertuples()]
    xx = np.arange(len(fs))
    axC.axhline(0.25, color="k", ls="--", lw=1, label="predicted 1/4")
    axC.plot(xx, fs.slope_cond, "o", color="C0", label="cond fit")
    axC.plot(xx, fs.slope_steps, "s", color="C1", label="steps fit")
    axC.set_xticks(xx)
    axC.set_xticklabels(labels, rotation=60, fontsize=6, ha="right")
    axC.set_ylabel(r"fitted log-log slope of $\lambda^\star$ vs $d/n$")
    axC.set_title("Fitted vs predicted rate", fontsize=9)
    axC.legend(fontsize=7)
    axC.grid(alpha=0.3)

    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"damping_theory_validation.{ext}", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
