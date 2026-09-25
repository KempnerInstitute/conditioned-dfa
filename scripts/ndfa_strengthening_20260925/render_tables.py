"""Render the correction tables from explicit, immutable cohort exports."""
from pathlib import Path
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "assets/ndfa_strengthening_20260925"
REGIMES = [("nuisance_dominant", "Nuisance-dominant"), ("low_sample_noisy", "Low-sample/noisy"),
           ("mixed_context", "Mixed-context"), ("task_aligned", "Task-aligned")]


def table(columns, header, rows, caption, label):
    return (r"\begin{table}[!htbp]" + "\n" + r"\centering\small" + "\n" +
            r"\setlength{\tabcolsep}{4pt}" + "\n" + r"\begin{tabular}{@{}" + columns + "@{}}\n" +
            r"\toprule" + "\n" + header + r" \\" + "\n" + r"\midrule" + "\n" +
            "\n".join(row + r" \\" for row in rows) + "\n" + r"\bottomrule" + "\n" +
            r"\end{tabular}" + "\n" + r"\caption{" + caption + "}\n" + r"\label{" + label + "}\n" +
            r"\end{table}" + "\n")


def main():
    d = pd.read_csv(DATA / "synthetic_stability.csv").set_index(["condition", "method"])
    rows = []
    for key, name in REGIMES:
        g = d.loc[key]
        rows.append(name + " & " + " & ".join(f"{g.loc[m,'median_loss']:,.2f}" for m in ["bp", "dfa_random", "ndfa_random"]) + f" & {100*g.loc['dfa_random','loss_above_10']:.1f}")
    stability = table("lrrrr", r"Regime & BP loss & DFA loss & A loss & DFA loss $>10$ (\%)", rows,
        r"\textbf{Predictive instability in the shared-rate synthetic sweep.} Median final training loss and the fraction of raw-DFA runs with loss above 10. BP uses 160 runs per regime; local rules use 480 crossed data/feedback runs. These distribution summaries do not treat crossed runs as independent replications. No activity-nDFA run exceeds the threshold. Large accuracy differences in this sweep therefore combine stabilization with directional effects; norm-matched controls address the latter separately.",
        "tab:infodfa_synthetic_stability")
    d = pd.read_csv(DATA / "adam_seed_summary.csv").set_index(["cell", "method"])
    rows = []
    for key,name in [("nuisance_hard","Nuisance"),("low_sample_noisy","Low-sample"),("mixed_hard","Mixed"),("clean_aligned","Task-aligned")]:
        g=d.loc[key]
        rows.append(name+" & "+" & ".join(f"${g.loc[m,'accuracy_pct']:.2f}\\pm{g.loc[m,'sem_pp']:.2f}$" for m in ["dfa_sgd","dfa_adam_hidden","dfa_diag_activity_sqrt","ndfa_activity"]))
    adam = table("lrrrr", r"Cell & DFA & Hidden Adam & Diagonal $C_A^{-1/2}$ & Activity nDFA", rows,
        r"\textbf{Exploratory optimizer and diagonal controls.} Test accuracy (\%), mean $\pm$ SEM over five model seeds after averaging two feedback draws per seed. These are single cells with configurations selected on test accuracy, not a validation-selected optimizer benchmark. The diagonal rule uses inverse square roots while activity nDFA uses an inverse; this comparison does not isolate off-diagonal structure at a common power. Adam adapts hidden-weight updates. The original settings and explicit cohort summaries accompany the reproduction files.",
        "tab:infodfa_adam_control")
    d = pd.read_csv(DATA / "bn_headtohead.csv").set_index("condition")
    rows=[]
    for key,name in REGIMES:
        g=d.loc[key]
        rows.append(f"{name} & {g.dfa_bn_pct:.2f} & {g.a_bn_pct:.2f} & ${g.delta_pp:+.2f}$ & $[{g.t95_low_pp:+.2f},{g.t95_high_pp:+.2f}]$")
    bn=table("lrrrr",r"Regime & DFA+BN & A+BN & $\Delta$ (pp) & Individual 95\% CI",rows,
        r"\textbf{Conditioning within a shared BatchNorm recipe.} Regime means of test accuracy (\%). Both rules use BN before the hidden ReLU under the synthetic sweep's shared learning rate and 14-epoch horizon. Contrasts average all 32 designed cells and three feedback draws within each of five global seeds; intervals use Student $t$ with four degrees of freedom. Conditioning adds a benefit in the nuisance and low-sample regimes but reduces accuracy in the other two. This comparison does not establish the ordering after independent learning-rate tuning or convergence.",
        "tab:infodfa_bn_baseline")
    rows=[]
    for stem,name in [("dfa_stall_threefactor","MNIST/tanh"),("dfa_stall_fashion_threefactor","Fashion/tanh"),("dfa_relu_mnist_threefactor","MNIST/ReLU")]:
        for cohort in ["extension","pooled"]:
            w=pd.read_csv(DATA/f"{stem}_{cohort}_seed_means.csv").pivot(index="seed",columns="method",values="test_acc")
            e=100*(w.endfa-w.dfa); k=100*(w.kndfa-w.ndfa)
            assert (e>0).all() and (k>0).all()
            p=stats.wilcoxon(e,method="exact").pvalue
            assert p==stats.wilcoxon(k,method="exact").pvalue
            seedrange=f"{w.index.min()}--{w.index.max()}"
            rows.append(f"{name} & {cohort.capitalize()} & {seedrange} & {len(w)} & {e.mean():+.2f} & {k.mean():+.2f} & {p:.5f}")
    extension=table("lllrrrr",r"Setting & Cohort & Seeds & $n$ & E$-$DFA & K$-$A & Nominal $p$",rows,
        r"\textbf{Post-hoc extensions of the frozen factor protocols.} Mean paired accuracy differences in percentage points. Each model/data-order seed averages three feedback draws. Every seed favors E over DFA and K over A; the last column is the identical exact two-sided Wilcoxon value for the two contrasts, without multiplicity adjustment. Pooled results include the original cohort and the post-hoc extension; they are descriptive and do not turn the original MNIST cohort into a test-blind confirmation. Explicit original, extension and pooled exports prevent the larger cohorts from silently replacing the original tables and figures.",
        "tab:factor_seed_extensions")
    for name,content in [("stability",stability),("adam",adam),("bn",bn),("extensions",extension)]:
        (DATA/f"table_{name}.tex").write_text(content)
    print("Rendered four tables from explicit seed-level evidence.")


if __name__ == "__main__":
    main()
