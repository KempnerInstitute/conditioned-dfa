"""Validate Proposition 3 (mode-timing law) in the aligned linear-Gaussian model.

Model (Appendix A notation): x ~ N(0, Sigma), scalar teacher y = w*.x + xi,
xi ~ N(0, sigma^2), n training samples, student w = (W2 W1)^T with fixed
aligned feedback B = alpha W2^T. Gradient flow on the empirical loss gives,
in the eigenbasis of the empirical second moment Sigma_hat,

    w_i(t) = w_i^LS (1 - exp(-r_i t)),   w_i^LS = w_i^* + nu_i,
    E[nu_i^2] = sigma^2 / (n lambda_hat_i),

with per-mode rates
    BP / DFA : r_i = eta lambda_hat_i
    nDFA     : r_i = eta lambda_hat_i / (lambda_hat_i + lambda_C).

All rules share the destination w^LS: conditioning changes only the clock.
Expected excess test risk (population-spectrum approximation, Advani & Saxe
2020 style):

    E R(t) = sum_i lambda_i w_i*^2 exp(-2 r_i t)
           + (sigma^2/n) sum_i (1 - exp(-r_i t))^2.

Checks performed here (all exact finite-n, no population approximation in the
simulation itself):
  1. Per-mode timing: t_task/t_nuis = kappa for BP/DFA vs
     rho_c = [lam_N (lam_T + lam_C)] / [lam_T (lam_N + lam_C)] for nDFA.
  2. Nuisance-dominant regime (task on lowest-lambda directions): the
     best-achievable test risk along the trajectory improves under
     conditioning, monotonically as lambda_C decreases; raw DFA/BP pay the
     full nuisance noise floor sigma^2 d_N / n before the task is fit.
  3. Task-aligned control (task on highest-lambda directions): conditioning
     does not improve the trajectory minimum and mild damping delays task
     fitting at matched step size (clean-control reversal).
  4. Crossing sweep: moving the task mode through the spectrum flips the sign
     of the conditioning gain; the two-block theory predicts the flip at
     lambda_task = lambda_nuis, and for a full spectrum the measured crossing
     position depends on sigma and lambda_C (noise modes faster than the task
     favor conditioning, slower ones penalize it).
  5. Two-block closed form: Delta ~= min(S,N)^2/(S+N) at strong compression,
     N = sigma^2 d_N / n.
Both the exact ODE solution of the empirical-loss gradient flow AND minibatch
SGD on an explicit two-layer linear network (W1 trained, W2 fixed, B = W2^T)
are simulated. The preconditioner uses the full-train Sigma_hat + lambda_C I
(estimation noise in the preconditioner is Proposition 2's subject, held
fixed here to isolate timing).

CPU-only; ~1 minute. Outputs (results/infodfa_mode_timing_v1/):
    mode_timing_trajectories.csv, mode_timing_sweep.csv,
    mode_timing_permode.csv, mode_timing_twoblock_check.csv,
    mode_timing_validation.pdf/.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "infodfa_mode_timing_v1"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- house style
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9.0,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.2,
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "lines.markersize": 4.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 400,
    }
)
COLORS = {"BP": "#0072B2", "DFA": "#999999", "nDFA": "#009E73", "dark": "#222222"}
DFA_DASH = (0, (4, 2))
# nDFA shades: light -> dark green as lambda_C decreases (stronger conditioning)
NDFA_SHADES = {1.0: "#7FCDBB", 0.1: "#009E73", 0.01: "#00543E"}

# ---------------------------------------------------------------- model setup
D = 32
KAPPA = 50.0
LAMBDAS = np.logspace(0.0, -np.log10(KAPPA), D)  # lambda_max = 1 .. lambda_min = 1/kappa
SIGMA_NOISE = 0.2  # label-noise std
N_TRAIN = 128
S_SIGNAL = 0.02  # population signal power S = sum_task lambda_i w_i*^2
LAMBDA_CS = [0.01, 0.1, 1.0]
N_SEEDS = 20
ETA = 1.0  # gradient-flow time unit; trajectory minima are invariant to it
T_GRID = np.concatenate([[0.0], np.logspace(-3, 3.6, 900)])
RNG_ROOT = np.random.SeedSequence(20260710)


def make_wstar(task_idx: np.ndarray) -> np.ndarray:
    """Teacher supported on task_idx (eigenbasis coords), signal power S_SIGNAL."""
    w = np.zeros(D)
    per_mode = S_SIGNAL / len(task_idx)  # split signal power evenly
    w[task_idx] = np.sqrt(per_mode / LAMBDAS[task_idx])
    return w


def rates(lams: np.ndarray, rule: str, lam_c: float = 0.0) -> np.ndarray:
    if rule in ("BP", "DFA"):
        return ETA * lams
    if rule == "nDFA":
        return ETA * lams / (lams + lam_c)
    raise ValueError(rule)


def theory_risk(t: np.ndarray, wstar_eig: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Population-spectrum approximation of E_xi R(t)."""
    decay = np.exp(-np.outer(t, r))  # (T, D)
    sig = decay**2 @ (LAMBDAS * wstar_eig**2)
    noi = (SIGMA_NOISE**2 / N_TRAIN) * ((1.0 - decay) ** 2).sum(axis=1)
    return sig + noi


def ode_risk(t: np.ndarray, wstar: np.ndarray, rule: str, lam_c: float,
             X: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Exact risk trajectory of gradient flow on the empirical loss.

    w(t) = V b(t), b(t) = b_LS (1 - exp(-r_hat t)) in Sigma_hat's eigenbasis.
    Risk uses the true Sigma = diag(LAMBDAS) (inputs are drawn in the
    eigenbasis, so Sigma is diagonal WLOG).
    """
    n = X.shape[0]
    Sig_hat = X.T @ X / n
    lh, V = np.linalg.eigh(Sig_hat)
    lh = np.clip(lh, 1e-12, None)
    r = rates(lh, rule, lam_c)
    b_ls = V.T @ wstar + (V.T @ (X.T @ xi / n)) / lh  # w_LS in eigenbasis
    b_star = V.T @ wstar
    decay = np.exp(-np.outer(t, r))  # (T, D)
    db = b_ls * (1.0 - decay) - b_star  # (T, D)
    G = V.T @ (LAMBDAS[:, None] * V)  # Sigma in Sigma_hat's eigenbasis
    return np.einsum("td,de,te->t", db, G, db)


def sgd_risk(wstar: np.ndarray, rule: str, lam_c: float, X: np.ndarray,
             xi: np.ndarray, rng: np.random.Generator, eta: float = 0.05,
             batch: int = 16, steps: int = 60000, n_hidden: int = 64):
    """Minibatch SGD on an explicit two-layer linear net, W2 fixed, B = W2^T.

    First-layer update: DFA outer product (B e) x^T, optionally
    right-multiplied by (Sigma_hat + lambda_C I)^{-1} (nDFA). With aligned
    B = W2^T the BP and DFA first-layer updates coincide exactly.
    Returns (effective_time, risk) at recorded steps; t_eff = step * eta *
    ||W2||^2 matches the gradient-flow clock (ETA = 1).
    """
    n = X.shape[0]
    y = X @ wstar + xi
    W2 = rng.standard_normal((1, n_hidden))
    W2 /= np.linalg.norm(W2)  # ||W2|| = 1 -> effective eta = eta
    B = W2.T  # aligned feedback, alpha = 1
    W1 = np.zeros((n_hidden, D))
    P = np.eye(D)
    if rule == "nDFA":
        P = np.linalg.inv(X.T @ X / n + lam_c * np.eye(D))
    # log-spaced recording so early minima (t ~ 1) are resolved
    rec = np.unique(np.concatenate(
        [[0], np.logspace(0, np.log10(steps), 60).astype(int)]))
    rec_set = set(rec.tolist())
    ts, risks = [], []
    for s in range(steps + 1):
        if s in rec_set:
            w_eff = (W2 @ W1).ravel()
            ts.append(s * eta)
            risks.append(float(((w_eff - wstar) ** 2 * LAMBDAS).sum()))
        idx = rng.integers(0, n, size=batch)
        xb, yb = X[idx], y[idx]
        e = xb @ (W2 @ W1).ravel() - yb  # (batch,)
        G = B @ (e[None, :] @ xb) / batch  # (n_hidden, D) DFA outer product
        W1 -= eta * G @ P
    return np.array(ts), np.array(risks)


def draw_data(rng: np.random.Generator):
    X = rng.standard_normal((N_TRAIN, D)) * np.sqrt(LAMBDAS)[None, :]
    xi = rng.standard_normal(N_TRAIN) * SIGMA_NOISE
    return X, xi


# ---------------------------------------------------------------- experiments
def run_regime(task_idx: np.ndarray, seeds) -> dict:
    """ODE risk trajectories (mean/sem over seeds) for BP/DFA/nDFA."""
    wstar = make_wstar(task_idx)
    out = {}
    rules = [("BP", 0.0), ("DFA", 0.0)] + [("nDFA", lc) for lc in LAMBDA_CS]
    for rule, lc in rules:
        curves = []
        for ss in seeds:
            rng = np.random.default_rng(ss)
            X, xi = draw_data(rng)
            curves.append(ode_risk(T_GRID, wstar, rule, lc, X, xi))
        curves = np.array(curves)
        key = rule if rule != "nDFA" else f"nDFA_{lc}"
        out[key] = {
            "mean": curves.mean(0),
            "sem": curves.std(0, ddof=1) / np.sqrt(len(seeds)),
            "rule": rule,
            "lambda_C": lc,
            "wstar": wstar,
        }
    return out


def best_along(mean_curve: np.ndarray) -> tuple[float, float]:
    i = int(np.argmin(mean_curve))
    return float(mean_curve[i]), float(T_GRID[i])


print("=== Proposition 3 validation ===")
seeds = RNG_ROOT.generate_state(N_SEEDS) % (2**31)

TASK_LOW = np.array([D - 2, D - 1])   # two lowest-variance directions
TASK_HIGH = np.array([0, 1])          # two highest-variance directions (control)

# --- check 1: per-mode timing ratios (exact algebra, recorded for the CSV)
lam_T, lam_N = LAMBDAS[-1], LAMBDAS[0]
permode_rows = []
for rule, lc in [("DFA", 0.0)] + [("nDFA", lc) for lc in LAMBDA_CS]:
    r = rates(LAMBDAS, rule, lc)
    ratio = r[0] / r[-1]  # fastest(nuis) / slowest(task) rate = t_task/t_nuis
    pred = (lam_N / lam_T if rule == "DFA"
            else (lam_N * (lam_T + lc)) / (lam_T * (lam_N + lc)))
    permode_rows.append(dict(rule=rule, lambda_C=lc, t_task_over_t_nuis=ratio,
                             predicted=pred))
    for i, (lam, ri) in enumerate(zip(LAMBDAS, r)):
        permode_rows.append(dict(rule=rule, lambda_C=lc, mode=i, eigenvalue=lam,
                                 rate=ri, t_fit_90=np.log(10.0) / ri))
pd.DataFrame(permode_rows).to_csv(OUT / "mode_timing_permode.csv", index=False)
print("timing ratios (t_task/t_nuis):",
      {f"{r['rule']}@{r['lambda_C']}": round(r["t_task_over_t_nuis"], 2)
       for r in permode_rows if "t_task_over_t_nuis" in r and not np.isnan(
           r.get("t_task_over_t_nuis", np.nan))})

# --- checks 2-3: trajectories in the two regimes
res_low = run_regime(TASK_LOW, seeds)
res_high = run_regime(TASK_HIGH, seeds)

traj_rows = []
for regime, res in [("nuisance_dominant", res_low), ("task_aligned", res_high)]:
    for key, v in res.items():
        best, tbest = best_along(v["mean"])
        print(f"[{regime}] {key:10s} R_dagger={best:.5f} at t={tbest:9.2f} "
              f"final={v['mean'][-1]:.5f}")
        for t, m, s in zip(T_GRID, v["mean"], v["sem"]):
            traj_rows.append(dict(regime=regime, rule=key, t=t, risk_mean=m,
                                  risk_sem=s))
pd.DataFrame(traj_rows).to_csv(OUT / "mode_timing_trajectories.csv", index=False)

noise_floor_nuis = SIGMA_NOISE**2 * (D - 2) / N_TRAIN
print(f"S={S_SIGNAL}, nuisance noise floor N=sigma^2 d_N/n={noise_floor_nuis:.5f}, "
      f"final risk sigma^2 d/n={SIGMA_NOISE**2 * D / N_TRAIN:.5f}")

# monotonicity check (Prop 3 iii): R_dagger decreasing in lambda_C strength
b_dfa = best_along(res_low["DFA"]["mean"])[0]
b_ndfa = {lc: best_along(res_low[f"nDFA_{lc}"]["mean"])[0] for lc in LAMBDA_CS}
ok_low = all(b_ndfa[lc] < b_dfa for lc in LAMBDA_CS) and (
    b_ndfa[0.01] < b_ndfa[0.1] < b_ndfa[1.0])
b_dfa_hi = best_along(res_high["DFA"]["mean"])[0]
b_ndfa_hi = {lc: best_along(res_high[f"nDFA_{lc}"]["mean"])[0] for lc in LAMBDA_CS}
ok_high = all(b_ndfa_hi[lc] >= b_dfa_hi * 0.999 for lc in LAMBDA_CS)
print(f"nuisance-dominant: conditioning improves R_dagger, monotone in "
      f"lambda_C: {ok_low}")
print(f"task-aligned control: conditioning does not improve R_dagger "
      f"(reversal): {ok_high}")

# task-fitting delay at matched eta in the control (Prop 3, part b)
for lc in LAMBDA_CS:
    delay = (LAMBDAS[0] + lc) / LAMBDAS[0]
    print(f"  matched-eta task-fit delay, lambda_C={lc}: x{delay:.2f}")

# material-improvement condition (Prop 3 iii, quantitative form):
# retained nuisance fraction at the stopping point ~ u*^rho_c with
# u* = Ntot/(S+Ntot); predicted Delta ~ N u*^rho_c (2 - u*^rho_c).
Ntot = SIGMA_NOISE**2 * D / N_TRAIN
u_star = Ntot / (S_SIGNAL + Ntot)
print("material-improvement condition (population two-block prediction "
      "vs measured, nuisance-dominant):")
for lc in LAMBDA_CS:
    rho_c = (LAMBDAS[0] * (LAMBDAS[-1] + lc)) / (LAMBDAS[-1] * (LAMBDAS[0] + lc))
    retained = u_star**rho_c
    delta_pred = noise_floor_nuis * retained * (2 - retained)
    delta_meas = b_dfa - b_ndfa[lc]
    print(f"  lambda_C={lc:5g} rho_c={rho_c:6.2f} retained u*^rho_c={retained:.3f}"
          f"  Delta_pred~{delta_pred:.5f}  Delta_meas={delta_meas:.5f}")

# corollary: separation time ~ fitting time of the fastest nuisance mode
for lc in LAMBDA_CS:
    diff = np.abs(res_low[f"nDFA_{lc}"]["mean"] - res_low["DFA"]["mean"])
    rel = diff / np.maximum(res_low["DFA"]["mean"], 1e-12)
    t_fast = 1.0 / (ETA * LAMBDAS[0])
    if np.any(rel > 0.05):
        i_sep = int(np.argmax(rel > 0.05))
        print(f"  separation time (5% rel. gap) lambda_C={lc}: "
              f"t={T_GRID[i_sep]:.2f}  (fastest-nuisance-mode time "
              f"1/(eta lambda_max)={t_fast:.2f})")
    else:
        print(f"  separation time (5% rel. gap) lambda_C={lc}: no 5% gap "
              f"(heavy damping reverts to a rescaled raw-DFA trajectory)")

# --- minibatch SGD spot check (nuisance-dominant, subset of seeds)
sgd_rows = []
wstar_low = make_wstar(TASK_LOW)
for rule, lc in [("DFA", 0.0), ("nDFA", 0.1), ("nDFA", 0.01)]:
    curves = []
    for ss in seeds[:8]:
        rng = np.random.default_rng(ss)
        X, xi = draw_data(rng)
        ts, risks = sgd_risk(wstar_low, rule, lc, X, xi,
                             np.random.default_rng(int(ss) + 1))
        curves.append(risks)
    curves = np.array(curves)
    key = rule if rule == "DFA" else f"nDFA_{lc}"
    for t, m, s in zip(ts, curves.mean(0),
                       curves.std(0, ddof=1) / np.sqrt(curves.shape[0])):
        sgd_rows.append(dict(regime="nuisance_dominant", rule=key, t=t,
                             risk_mean=m, risk_sem=s))
sgd_df = pd.DataFrame(sgd_rows)
sgd_df.to_csv(OUT / "mode_timing_sgd.csv", index=False)
for key in sgd_df.rule.unique():
    sub = sgd_df[sgd_df.rule == key]
    print(f"[SGD] {key:10s} min risk {sub.risk_mean.min():.5f} "
          f"(ODE {best_along(res_low[key]['mean'])[0]:.5f})")

# --- check 4: crossing sweep (single task mode moved through the spectrum)
sweep_rows = []
sweep_seeds = seeds[:10]
for sigma_sweep in [0.1, 0.2, 0.4]:
    for k in range(D):
        wstar = np.zeros(D)
        wstar[k] = np.sqrt(S_SIGNAL / LAMBDAS[k])
        rvals = {}
        for rule, lc in [("DFA", 0.0)] + [("nDFA", lc) for lc in LAMBDA_CS]:
            curves = []
            for ss in sweep_seeds:
                rng = np.random.default_rng(int(ss) + 7)
                X = rng.standard_normal((N_TRAIN, D)) * np.sqrt(LAMBDAS)[None, :]
                xi = rng.standard_normal(N_TRAIN) * sigma_sweep
                curves.append(ode_risk(T_GRID, wstar, rule, lc, X, xi))
            key = rule if rule == "DFA" else f"nDFA_{lc}"
            rvals[key] = best_along(np.mean(curves, axis=0))[0]
        for lc in LAMBDA_CS:
            sweep_rows.append(dict(
                sigma=sigma_sweep, task_mode=k, lambda_task=LAMBDAS[k],
                lambda_C=lc, R_dagger_DFA=rvals["DFA"],
                R_dagger_nDFA=rvals[f"nDFA_{lc}"],
                delta_rel=(rvals["DFA"] - rvals[f"nDFA_{lc}"]) / rvals["DFA"]))
sweep_df = pd.DataFrame(sweep_rows)
sweep_df.to_csv(OUT / "mode_timing_sweep.csv", index=False)
for sigma_sweep in [0.1, 0.2, 0.4]:
    for lc in LAMBDA_CS:
        sub = sweep_df[(sweep_df.lambda_C == lc)
                       & (sweep_df.sigma == sigma_sweep)].sort_values("task_mode")
        sign = np.sign(sub.delta_rel.values)
        # crossing: last task position (from top) where conditioning hurts
        neg = np.where(sign < 0)[0]
        pos = np.where(sign > 0)[0]
        lam_cross = (np.nan if len(neg) == 0 or len(pos) == 0
                     else np.sqrt(sub.lambda_task.values[neg[-1]]
                                  * sub.lambda_task.values[neg[-1] + 1])
                     if neg[-1] + 1 < D else np.nan)
        print(f"[sweep sigma={sigma_sweep} lambda_C={lc}] "
              f"delta_rel(top)={sub.delta_rel.iloc[0]:+.3f} "
              f"delta_rel(bottom)={sub.delta_rel.iloc[-1]:+.3f} "
              f"crossing at lambda_task ~ {lam_cross:.3f}")

# --- check 5: two-block closed form in its stated limit (rho_raw -> inf,
# rho_c -> 1): R_raw = min(S, N + S NT/(S+NT)), R_c = S Ntot/(S+Ntot),
# Delta_exact = R_raw - R_c; leading order (NT << N): min(S,N)^2/(S+N).
tb_rows = []
d_T, d_N = 2, D - 2
Nn = SIGMA_NOISE**2 * d_N / N_TRAIN
NT = SIGMA_NOISE**2 * d_T / N_TRAIN
u = np.concatenate([[1.0], np.logspace(0, -10, 4000)])
for S in np.logspace(-3, 0, 13):
    def twoblock(rho):
        R = S * u**2 + NT * (1 - u) ** 2 + Nn * (1 - u**rho) ** 2
        return float(R.min())
    delta = twoblock(1e6) - twoblock(1.0)
    pred_exact = (min(S, Nn + S * NT / (S + NT))
                  - S * (Nn + NT) / (S + Nn + NT))
    pred_lead = min(S, Nn) ** 2 / (S + Nn)
    tb_rows.append(dict(S=S, N=Nn, delta_measured=delta,
                        delta_exact_formula=pred_exact,
                        delta_leading_order=pred_lead))
tb_df = pd.DataFrame(tb_rows)
tb_df.to_csv(OUT / "mode_timing_twoblock_check.csv", index=False)
rel_exact = np.abs(tb_df.delta_measured - tb_df.delta_exact_formula) / np.maximum(
    tb_df.delta_measured, 1e-12)
rel_lead = np.abs(tb_df.delta_measured - tb_df.delta_leading_order) / np.maximum(
    tb_df.delta_measured, 1e-12)
print(f"two-block Delta: exact formula max rel. deviation "
      f"{rel_exact.max():.3f}; leading-order median {np.median(rel_lead):.2f}")

# -------------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.6))
plt.subplots_adjust(left=0.055, right=0.995, top=0.82, bottom=0.19, wspace=0.34)

# Panel A: per-mode fitting curves
ax = axes[0]
tt = np.logspace(-2, 3.6, 400)
for rule, lc, color, ls in [("DFA", 0.0, COLORS["DFA"], DFA_DASH),
                            ("nDFA", 0.1, NDFA_SHADES[0.1], "-")]:
    r = rates(LAMBDAS, rule, lc)
    ax.plot(tt, 1 - np.exp(-r[0] * tt), color=color, ls=ls, lw=1.4, alpha=0.85)
    ax.plot(tt, 1 - np.exp(-r[-1] * tt), color=color, ls=ls, lw=2.4)
ax.set_xscale("log")
ax.set_xlabel("time $t$ (matched $\\eta$)")
ax.set_ylabel("mode fraction fit")
ax.text(0.98, 0.05, "nuisance $\\lambda_{\\max}$: thin\ntask $\\lambda_{\\min}$: thick",
        transform=ax.transAxes, fontsize=6.6, color=COLORS["dark"],
        ha="right", va="bottom")
ax.annotate("$t_{task}/t_{nuis}=\\kappa=50$", xy=(0.35, 0.30),
            xycoords="axes fraction", fontsize=7.0, color=COLORS["DFA"])
rho_c = lam_N * (lam_T + 0.1) / (lam_T * (lam_N + 0.1))
ax.annotate(f"$\\rho_c={rho_c:.1f}$", xy=(0.62, 0.62), xycoords="axes fraction",
            fontsize=7.0, color=NDFA_SHADES[0.1])
ax.set_title("A  per-mode timing", loc="left", fontweight="bold")

# Panels B, C: risk trajectories
for ax, res, regime, title in [
    (axes[1], res_low, "nuisance_dominant", "B  task on low-$\\lambda$"),
    (axes[2], res_high, "task_aligned", "C  task on high-$\\lambda$ (control)"),
]:
    order = ["BP", "DFA"] + [f"nDFA_{lc}" for lc in LAMBDA_CS]
    for key in order:
        v = res[key]
        if key == "BP":
            color, ls, lw, lab = COLORS["BP"], "-", 2.4, "BP"
        elif key == "DFA":
            color, ls, lw, lab = COLORS["DFA"], DFA_DASH, 2.0, "DFA"
        else:
            lc = v["lambda_C"]
            color, ls, lw = NDFA_SHADES[lc], "-", 1.8
            lab = f"nDFA $\\lambda_C{{=}}{lc:g}$"
        m = v["mean"]
        ax.plot(T_GRID[1:], m[1:], color=color, ls=ls, lw=lw, label=lab)
        ax.fill_between(T_GRID[1:], (m - v["sem"])[1:], (m + v["sem"])[1:],
                        color=color, alpha=0.16, lw=0)
        i = int(np.argmin(m))
        ax.plot([T_GRID[i]], [m[i]], "o", color=color, ms=4.0, zorder=5)
    if regime == "nuisance_dominant":
        sub0 = sgd_df[sgd_df.rule == "DFA"]
        ax.plot(sub0.t.values[1:], sub0.risk_mean.values[1:], marker="s", ms=2.2,
                lw=0, color=COLORS["DFA"], alpha=0.55)
        sub1 = sgd_df[sgd_df.rule == "nDFA_0.1"]
        ax.plot(sub1.t.values[1:], sub1.risk_mean.values[1:], marker="s", ms=2.2,
                lw=0, color=NDFA_SHADES[0.1], alpha=0.55)
        ax.text(0.02, 0.03, "squares: minibatch SGD", transform=ax.transAxes,
                fontsize=6.4, color=COLORS["dark"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("time $t$ (matched $\\eta$)")
    ax.set_ylabel("test risk $\\mathbb{E}R(t)$")
    ax.set_title(title, loc="left", fontweight="bold")
    if regime == "nuisance_dominant":
        ax.legend(frameon=False, fontsize=6.0, loc="upper right",
                  handlelength=1.5, labelspacing=0.25, borderaxespad=0.1)

# Panel D: crossing sweep
ax = axes[3]
for lc in LAMBDA_CS:
    sub = sweep_df[(sweep_df.lambda_C == lc)
                   & (sweep_df.sigma == SIGMA_NOISE)].sort_values("lambda_task")
    ax.plot(sub.lambda_task, 100 * sub.delta_rel, color=NDFA_SHADES[lc], lw=1.8,
            label=f"$\\lambda_C{{=}}{lc:g}$")
sub = sweep_df[(sweep_df.lambda_C == 0.1)
               & (sweep_df.sigma == 0.4)].sort_values("lambda_task")
ax.plot(sub.lambda_task, 100 * sub.delta_rel, color=NDFA_SHADES[0.1], lw=1.2,
        ls=(0, (1, 1)), label="$\\lambda_C{=}0.1,\\,\\sigma{=}0.4$")
ax.axhline(0, color=COLORS["dark"], lw=0.8)
ax.set_xscale("log")
ax.set_xlabel("$\\lambda_{task}$ (task-mode eigenvalue)")
ax.set_ylabel("$\\Delta R^{\\dagger}/R^{\\dagger}_{DFA}$ (%)")
ax.legend(frameon=False, fontsize=6.4, loc="lower left")
ax.set_title("D  crossing condition", loc="left", fontweight="bold")

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"mode_timing_validation.{ext}")
print(f"wrote figure + CSVs to {OUT}")
