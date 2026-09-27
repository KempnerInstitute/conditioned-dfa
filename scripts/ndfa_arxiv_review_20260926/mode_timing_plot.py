"""Plot the archived mode-timing tables without rerunning simulations."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
DATA=Path(__file__).resolve().parents[2]/'assets/ndfa_arxiv_review_20260926'
COLORS={"BP":"#0072B2","DFA":"#999999","nDFA":"#009E73","dark":"#222222"}
DFA_DASH=(0,(4,2));NDFA_SHADES={1.0:"#7FCDBB",.1:"#009E73",.01:"#00543E"}
LAMBDA_AS=[.01,.1,1.0];SIGMA_NOISE=.2
LAMBDAS=np.logspace(0,-np.log10(50),32);lam_N=LAMBDAS[0];lam_T=LAMBDAS[-1]
def rates(lambdas,rule,la):return lambdas if rule in ['DFA','BP'] else lambdas/(lambdas+la)
frame=pd.read_csv(DATA/'mode_timing_trajectories.csv')
def results(regime):
    res={}
    for rule,g in frame[frame.regime==regime].groupby('rule',sort=False):
        res[rule]={'mean':g.risk_mean.to_numpy(),'sem':g.risk_sem.to_numpy(),'lambda_A':float(rule.split('_')[1]) if '_' in rule else 0}
    return res
res_low=results('nuisance_dominant');res_high=results('task_aligned')
T_GRID=frame[(frame.regime=='nuisance_dominant')&(frame.rule=='BP')].t.to_numpy()
sgd_df=pd.read_csv(DATA/'mode_timing_sgd.csv');sweep_df=pd.read_csv(DATA/'mode_timing_sweep.csv')
fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.6))
plt.subplots_adjust(left=0.055, right=0.995, top=0.82, bottom=0.19, wspace=0.34)

# Panel A: per-mode fitting curves
ax = axes[0]
tt = np.logspace(-2, 3.6, 400)
for rule, la, color, ls in [("DFA", 0.0, COLORS["DFA"], DFA_DASH),
                            ("nDFA", 0.1, NDFA_SHADES[0.1], "-")]:
    r = rates(LAMBDAS, rule, la)
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
    order = ["BP", "DFA"] + [f"nDFA_{la}" for la in LAMBDA_AS]
    for key in order:
        v = res[key]
        if key == "BP":
            color, ls, lw, lab = COLORS["BP"], "-", 2.4, "BP"
        elif key == "DFA":
            color, ls, lw, lab = COLORS["DFA"], DFA_DASH, 2.0, "DFA"
        else:
            la = v["lambda_A"]
            color, ls, lw = NDFA_SHADES[la], "-", 1.8
            lab = f"nDFA $\\lambda_A{{=}}{la:g}$"
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
for la in LAMBDA_AS:
    sub = sweep_df[(sweep_df.lambda_A == la)
                   & (sweep_df.sigma == SIGMA_NOISE)].sort_values("lambda_task")
    ax.plot(sub.lambda_task, 100 * sub.delta_rel, color=NDFA_SHADES[la], lw=1.8,
            label=f"$\\lambda_A{{=}}{la:g}$")
sub = sweep_df[(sweep_df.lambda_A == 0.1)
               & (sweep_df.sigma == 0.4)].sort_values("lambda_task")
ax.plot(sub.lambda_task, 100 * sub.delta_rel, color=NDFA_SHADES[0.1], lw=1.2,
        ls=(0, (1, 1)), label="$\\lambda_A{=}0.1,\\,\\sigma{=}0.4$")
ax.axhline(0, color=COLORS["dark"], lw=0.8)
ax.set_xscale("log")
ax.set_xlabel("$\\lambda_{task}$ (task-mode eigenvalue)")
ax.set_ylabel("$\\Delta R^{\\dagger}/R^{\\dagger}_{DFA}$ (%)")
ax.legend(frameon=False, fontsize=6.4, loc="lower left")
ax.set_title("D  crossing condition", loc="left", fontweight="bold")

# Native two-row presentation keeps labels readable at manuscript width.
fig.set_size_inches(5.5, 4.6)
for i, axis in enumerate(axes):
    row, col = divmod(i, 2)
    x, y = .12 + .49*col, .61 if row == 0 else .21
    axis.set_position([x, y, .36, .28])
    axis.set_title('', loc='left')
    axis.set_title(['Mode timing', 'Task in low-variance modes',
                    'Task in high-variance modes', 'Task-location dependence'][i],
                   fontsize=8, fontweight='bold')
    axis.tick_params(labelsize=7, pad=2)
    axis.set_xlabel('Flow time' if i < 3 else 'Task eigenvalue', fontsize=8)
    axis.set_ylabel(['Fraction fitted', 'Expected excess test risk', 'Expected excess test risk',
                     'Relative risk gain (%)'][i], fontsize=8)
    fig.text(x-.09, y+.34, chr(65+i), fontsize=10, fontweight='bold', va='bottom')
    for item in axis.texts:
        item.set_fontsize(6.5)
for axis in axes[:2]:
    for item in list(axis.texts):item.remove()
handles, labels = axes[1].get_legend_handles_labels()
axes[1].get_legend().remove()
fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=7, frameon=False)
axes[1].set_yticks([.01,.015,.02]);axes[1].set_yticklabels(['0.010','0.015','0.020'])
axes[1].minorticks_off()
