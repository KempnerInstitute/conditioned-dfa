"""Compact four-panel figures at the final ICLR text width.

Use the corrected September 19 generator for all measurements and uncertainty
estimates. Figures 1, 2 and 4 use four-panel rows; Figure 3 is compacted and
Figure 5 includes the existing longer-work and retuned-width follow-ups.
The restored conditioning panel is an analytic identity, not a simulated
convergence-rate claim.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection, PolyCollection, LineCollection
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "corrected_figures", ROOT / "scripts/ndfa_revision_20260919/redraw.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
COL = previous.m.COL
NAMES = ["iclr_fig_theory_conditioning", "iclr_fig1_rule_and_positive_regimes",
         "iclr_fig_controls_composite"]
REPLICATION = "iclr_fig_error_kndfa_replication"
PRACTICAL = "ndfa_final_test_summary_20260915"
STABILITY = "ndfa_targeted_followups_20260918"


def measured_artists(fig):
    """Track plotted values independently of positions, labels and colors."""
    result = []
    for ax in fig.axes:
        row = {"lines": [], "collections": [], "bars": []}
        for line in ax.lines:
            row["lines"].append([np.asarray(line.get_xdata(), float).tolist(),
                                 np.asarray(line.get_ydata(), float).tolist()])
        for artist in ax.collections:
            if isinstance(artist, PathCollection):
                row["collections"].append(np.asarray(artist.get_offsets()).tolist())
            elif isinstance(artist, LineCollection):
                row["collections"].append([x.tolist() for x in artist.get_segments()])
            elif isinstance(artist, PolyCollection):
                row["collections"].append([x.vertices.tolist() for x in artist.get_paths()])
        for artist in ax.patches:
            if isinstance(artist, Rectangle):
                row["bars"].append([float(artist.get_x()), float(artist.get_y()),
                                    float(artist.get_width()), float(artist.get_height())])
        result.append(row)
    return json.dumps(result, sort_keys=True)


def row_layout(fig, titles, *, height=2.12, positions=None, bottom=.28, top=.79):
    fig.set_layout_engine(None)
    fig.set_size_inches(5.5, height)
    for item in list(fig.texts) + list(fig.legends):
        item.remove()
    if positions is None:
        positions = [(0.075 + .25*i, .169) for i in range(4)]
        positions[-1] = (.825, .157)
    for i, (ax, title, (left, width)) in enumerate(zip(fig.axes, titles, positions)):
        ax.set_position([left, bottom, width, top-bottom])
        for t in list(ax.texts):
            if t.get_text().strip() in list("ABCD"):
                t.remove()
        ax.set_title("", loc="left")
        ax.set_title("", loc="right")
        ax.set_title(title, fontsize=7.3, fontweight="normal", pad=5)
        ax.tick_params(labelsize=6.8, pad=2, width=.65, length=2.5)
        for label in [ax.xaxis.label, ax.yaxis.label]:
            label.set_size(7.2)
        ax.xaxis.labelpad = 3
        ax.yaxis.labelpad = 3
        for spine in ax.spines.values():
            spine.set_linewidth(.65)
        for line in ax.lines:
            if line.get_linewidth() > 1.6:
                line.set_linewidth(1.6)
            if line.get_markersize() > 4.5:
                line.set_markersize(4.5)
        # The letters share a row above all titles and sit left of axis labels.
        fig.text(.005 + .25*i, .93, chr(65+i), fontsize=9,
                 fontweight="bold", va="bottom")


def small_legend(ax, **kwargs):
    return ax.legend(frameon=False, fontsize=6.5, handlelength=1.4,
                     handletextpad=.4, borderaxespad=.15, labelspacing=.25,
                     **kwargs)


def theory():
    fig, axes = plt.subplots(1, 4)
    row_layout(fig, ["Task / nuisance", "Spectral weights", "Conditioning bound",
                     "Task orientation"], height=2.08,
               positions=[(.012, .202), (.318, .17), (.570, .17), (.823, .159)],
               bottom=.27, top=.78)
    a, b, c, d = axes
    a.axis("off")
    a.set_xlim(-1.25, 1.5)
    a.set_ylim(-1.0, 1.55)
    for width, alpha in [(2.55, .32), (1.92, .46)]:
        a.add_patch(Ellipse((0, 0), width, width*.28, facecolor="#B7C9BD",
                            edgecolor="none", alpha=alpha))
    # A nonzero residual in both coordinates; inverse conditioning nearly
    # recovers this direction. All arrows have a common origin.
    a.plot([0, .37], [0, 1.24], "--", color="#444444", lw=.9, zorder=2)
    for end, color in [((1.1, .20), COL["dfa"]), ((.30, 1.01), COL["a"])]:
        a.add_patch(FancyArrowPatch((0, 0), end, arrowstyle="-|>",
                                   mutation_scale=8, lw=1.6, color=color,
                                   shrinkA=0, shrinkB=0, zorder=3))
    a.text(-.45, 1.16, "Task", ha="center", fontsize=7, color="#444444")
    a.text(.49, .77, "nDFA", fontsize=7, color=COL["a"])
    a.text(.71, .34, "DFA", fontsize=7, color=COL["dfa"])
    a.text(.66, -.55, "Nuisance", ha="center", fontsize=7, color="#555555")

    lam = np.geomspace(1, 70, 12)
    gain = lam/(lam+.001)
    b.plot(range(1, 13), lam/lam.max(), "o--", color=COL["dfa"], ms=2.5,
           lw=1.4, label="BP / DFA")
    b.plot(range(1, 13), gain/gain.max(), "o-", color=COL["a"], ms=2.5,
           lw=1.4, label="nDFA")
    b.set(xlabel="Eigendirection", ylabel="Relative weight", ylim=(-.03, 1.10))
    b.set_xticks([1, 6, 12])
    b.set_yticks([0, .5, 1], ["0", "0.5", "1"])
    small_legend(b, loc="center left")

    kappa = np.array([1, 2, 5, 10, 20, 50, 100, 200.])
    rho = (kappa+.001)/(1+.001)
    c.loglog(kappa, rho, "o", color=COL["a"], ms=3, label="Analytic")
    c.loglog(kappa, kappa, "--", color="#333333", lw=.9, label="Bound")
    c.set(xlabel=r"Condition number $\kappa$", ylabel=r"Conditioning gain $\rho$")
    c.set_xticks([1, 10, 100])
    c.set_yticks([1, 10, 100])
    c.xaxis.set_minor_locator(NullLocator())
    c.yaxis.set_minor_locator(NullLocator())
    small_legend(c, loc="lower right")

    linear = pd.read_csv(previous.DATA / "linear_simulation.csv")
    for offset, method, label, color, marker in [
        (-.15, "dfa", "DFA", COL["dfa"], "o"),
        (.15, "activity", "nDFA", COL["a"], "s")]:
        values = linear[linear.rule == method].groupby("task_high").steps.mean()
        means = values.reindex([False, True]).to_numpy()
        d.vlines(np.arange(2)+offset, 10, means, color=color, lw=1)
        d.plot(np.arange(2)+offset, means, marker, color=color, ms=4,
               label=label, ls="none")
    d.set(yscale="log", ylim=(10, 14000), xlim=(-.45, 1.45),
          ylabel="Updates to target", xlabel="High variance in")
    d.set_xticks([0, 1], ["Nuisance", "Task"])
    d.set_yticks([10, 100, 1000, 10000])
    d.yaxis.set_minor_locator(NullLocator())
    small_legend(d, loc="upper right")
    for ax in [b, c, d]:
        ax.grid(axis="both" if ax != d else "y", color="#E8EAE6", lw=.5)
        ax.set_axisbelow(True)
    return fig


def positive(fig):
    row_layout(fig, ["Nuisance-dominant", "Low-sample / noisy", "Mixed-context", "Activity effect"],
               height=2.14, positions=[(.075,.177),(.314,.177),(.553,.177),(.825,.169)])
    for i, ax in enumerate(fig.axes[:3]):
        ax.set(xlabel="Epoch", ylabel="Test accuracy (%)" if i == 0 else "")
        ax.set_xticks([0, 7, 14])
        if i:
            ax.tick_params(labelleft=False)
        for line in ax.lines:
            if line.get_label() in ["BP", "BP (tuned)"] or len(line.get_xdata()) == 1:
                line.set_color(COL["bp"])
    fig.axes[3].set_xlabel("Gain (pp)")
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5,.015),
               ncol=5, frameon=False, fontsize=7, columnspacing=1.35,
               handlelength=1.8, handletextpad=.5)
    return fig


def controls(fig):
    row_layout(fig, ["Norm matching", "Nuisance loading", "BP conditioning", "Weight alignment"],
               height=2.16, bottom=.28, top=.79)
    a, b, c, d = fig.axes
    a.set_ylabel("Gain (pp)")
    small_legend(a, loc="upper center", bbox_to_anchor=(.5,-.27))
    b.set_xlabel("Nuisance / task energy", fontsize=6.8)
    b.set_ylabel("nDFA − DFA (pp)")
    b.set_xticks([1, 10, 100])
    b.xaxis.set_minor_locator(NullLocator())
    b.set_yticks([0, 30, 60])
    handles, labels = b.get_legend_handles_labels()
    b.legend(handles, ["Nuisance", "Low-N", "Mixed", "Task"], frameon=False,
             fontsize=6.4, loc="upper center", bbox_to_anchor=(.5,-.27),
             ncol=2, handlelength=.6, handletextpad=.35, columnspacing=.6,
             borderaxespad=.15, labelspacing=.25)
    for t in b.texts:
        t.set_fontsize(6.7)
        t.set_position((.06,.87))
        t.set_ha("left")
    c.set_ylabel("Accuracy (%)")
    c.set_ylim(0, 104)
    c.set_yticks([0, 50, 100])
    for patch in c.patches[:4]:
        patch.set_facecolor(COL["bp"])
    for t in list(c.texts):
        t.remove()  # Bar heights communicate the effect without redundant numbers.
    c.set_xticks(range(4), ["Nuis.", "Low-N", "Mixed", "Clean"], rotation=35, ha="right")
    small_legend(c, loc="upper center", bbox_to_anchor=(.5,-.27), ncol=1)
    d.set_ylabel("Alignment")
    d.set_xticks([0,50,100])
    d.set_yticks([0,.2,.4])
    for line in d.lines:
        if line.get_label() == "A":
            line.set_label("nDFA")
    small_legend(d, loc="upper center", bbox_to_anchor=(.5,-.27))
    return fig


def compact_replication(fig):
    """Preserve all three panels and every seed/interval; reduce empty height."""
    row_layout(fig, ["E − DFA", "K − A", "Predictive loss"], height=1.80,
               positions=[(.08,.225),(.417,.225),(.755,.225)], bottom=.255, top=.79)
    for i, letter in enumerate(fig.texts):
        letter.set_position((.005+i*.3375,.925))
    for ax in fig.axes[:2]:
        for item in list(ax.texts):
            item.remove()  # Means already have explicit markers and intervals.
        ax.set_ylabel("Gain (pp)")
    fig.axes[0].set_yticks([0,5,10])
    fig.axes[1].set_yticks([0,.5,1], ["0", "0.5", "1"])
    fig.axes[2].set_ylabel("Cross-entropy")
    fig.axes[2].set_yticks([.1,1,10,100,1000])
    fig.axes[2].yaxis.set_minor_locator(NullLocator())
    return fig


def practical_results():
    """Four questions, with all original comparisons and two existing follow-ups."""
    source = ROOT / "docs/research/ndfa_final_test_20260915/summary.json"
    summary = json.loads(source.read_text())
    assert summary["accepted"] and len(summary["contrasts"]) == 45
    endpoints = pd.read_csv(previous.DATA / "followup_endpoints.csv")
    endpoints = endpoints[~endpoints.development]
    contrasts = pd.read_csv(previous.DATA / "followup_contrasts.csv")
    fig, axes = plt.subplots(1,4)
    row_layout(fig, ["Matched work", "Longer work", "Transferred damping", "Retuned damping"],
               height=2.28, bottom=.34, top=.80,
               positions=[(.075,.174),(.312,.174),(.577,.169),(.832,.15)])
    a,b,c,d = axes
    records = []
    reference_colors = {"dfa":COL["a"], "bp":COL["bp"], "fd_dfa":"#AA4499"}

    def plot(ax, x, values, color, record, *, primary=False, marker="o"):
        values = np.asarray(values, float)
        mean = values.mean()
        half = t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))
        if "expected" in record:
            np.testing.assert_allclose([mean,mean-half,mean+half],record.pop("expected"),atol=1e-10,rtol=0)
        ax.scatter(x+np.linspace(-.034,.034,len(values)),values,s=7,alpha=.28,marker=marker,
                   color=color,edgecolors="none",zorder=2)
        ax.errorbar(x,mean,yerr=half,fmt="D" if primary else marker,
                    color=color,ms=3.3,lw=1,capsize=1.7,zorder=3)
        record.update(paired_values=values.tolist(),n=len(values),mean=float(mean),
                      ci95=[float(mean-half),float(mean+half)],primary=primary)
        records.append(record)

    # All five contrasts from the original matched-work panel remain present.
    for x, cohort in enumerate(["optimizer","forward"]):
        for offset, reference in [(-.18,"dfa"),(0,"bp"),(.18,"fd_dfa")]:
            if cohort == "optimizer" and reference == "fd_dfa":
                continue
            r = next(r for r in summary["contrasts"] if
                     (r["cohort"],r["first"],r["second"]) == (cohort,"ndfa",reference))
            v = r["accuracy"]
            assert len(v["paired_values"]) == 8
            plot(a,x+offset,v["paired_values"],reference_colors[reference],
                 {"panel":"A","cohort":cohort,"contrast":"ndfa - "+reference,
                  "expected":[v["mean"],*v["ci95"]]},primary=r["primary_accuracy"])
    a.set_xticks([0,1],["SGD","FD"])
    a.set(xlabel="30-s study",ylabel="Gain (pp)",xlim=(-.42,1.42),ylim=(-3.6,3.4))
    a.set_yticks([-3,0,3])

    def followup(ax,panel,phase,field,levels,comparisons):
        for x,level in enumerate(levels):
            p = endpoints[(endpoints.phase==phase)&(endpoints[field]==level)].pivot(
                index="seed",columns="method",values="test_accuracy")
            for offset,(first,second,color) in zip([-.18,0,.18],comparisons):
                values = 100*(p[first]-p[second])
                assert len(values)==5 and values.notna().all()
                r = contrasts[(contrasts.phase==phase)&(contrasts[field]==level)&
                              (contrasts.contrast==first+" - "+second)]
                assert len(r)==1
                r = r.iloc[0]
                plot(ax,x+offset,values.to_numpy(),color,
                     {"panel":panel,"cohort":phase,"level":level,"contrast":first+" - "+second,
                      "expected":[r["mean"],r.low,r.high]})
    followup(b,"B","work","budget",[60,120],
             [("ndfa","dfa",COL["a"]),("ndfa","bp",COL["bp"]),("ndfa","fd_dfa","#AA4499")])
    b.set_xticks([0,1],["60","120"])
    b.set(xlabel="Update work (s)",ylabel="",xlim=(-.42,1.42),ylim=a.get_ylim())
    b.set_yticks([-3,0,3]); b.tick_params(labelleft=False)

    # All four original transferred-damping contrasts remain, grouped by width.
    for x,width in enumerate([1024,2048]):
        for offset,n_train,marker in [(-.13,10000,"o"),(.13,48000,"s")]:
            r = next(r for r in summary["contrasts"] if
                     (r["cohort"],r["first"],r["second"],r["n_train"],r["width"]) ==
                     ("factor","ndfa","kndfa",n_train,width))
            v = r["accuracy"]; values = -np.asarray(v["paired_values"])
            assert len(values)==8
            plot(c,x+offset,values,COL["k"],
                 {"panel":"C","cohort":"factor","width":width,"n_train":n_train,
                  "contrast":"kndfa - ndfa","expected":[-v["mean"],-v["ci95"][1],-v["ci95"][0]]},marker=marker)
    c.set_xticks([0,1],["3.68","8.41"])
    c.set(xlabel="Parameters (M)",ylabel="K − A (pp)",xlim=(-.37,1.37),ylim=(-3.6,3.4))
    c.set_yticks([-3,0,3])

    followup(d,"D","width","width",[1024,2048],
             [("ndfa","dfa",COL["a"]),("endfa","dfa",COL["e"]),("kndfa","ndfa",COL["k"])])
    d.set_xticks([0,1],["1024","2048"])
    d.set(xlabel="Hidden width",ylabel="Gain (pp)",xlim=(-.42,1.42),ylim=(-1.6,9.2))
    d.set_yticks([0,4,8])

    for ax in axes:
        ax.axhline(0,color=".6",lw=.65,ls=":",zorder=1)
        ax.grid(axis="y",color=".93",lw=.5)
        ax.set_axisbelow(True)
    handles = [Line2D([],[],color=color,marker="o",lw=0,ms=3) for color in reference_colors.values()]
    fig.legend(handles,["A − DFA","A − BP","A − FD"],frameon=False,fontsize=6.5,
               ncol=3,loc="lower center",bbox_to_anchor=(.285,.065),
               handlelength=.7,handletextpad=.35,columnspacing=.9)
    fig.legend([Line2D([],[],color=COL["k"],marker=m,lw=0,ms=3) for m in ["o","s"]],
               ["10k examples","48k examples"],frameon=False,fontsize=6.3,ncol=1,
               loc="lower center",bbox_to_anchor=(.661,.03),handlelength=.7,
               handletextpad=.35,labelspacing=.25)
    fig.legend([Line2D([],[],color=COL[k],marker="o",lw=0,ms=3) for k in ["a","e","k"]],
               ["A − DFA","E − DFA","K − A"],frameon=False,fontsize=6.3,ncol=1,
               loc="lower center",bbox_to_anchor=(.918,.01),handlelength=.7,
               handletextpad=.35,labelspacing=.20)
    assert len(records)==21 and sum(r["primary"] for r in records)==2
    for record in records:
        lo,hi = axes["ABCD".index(record["panel"])].get_ylim()
        assert lo < min(record["paired_values"]+record["ci95"])
        assert hi > max(record["paired_values"]+record["ci95"])
    return fig,{"contrasts":records,"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                "all_nine_original_contrasts_retained":True,"no_cohort_pooling":True,
                "followup_contrasts_checked_against_saved_summaries":12}


def stability_only(fig):
    """The other two follow-up panels now appear in the main Figure 5."""
    before = json.loads(measured_artists(fig))[2]
    for ax in list(fig.axes[:2]):
        fig.delaxes(ax)
    row_layout(fig,["Normalization and predictive stability"],height=1.85,
               positions=[(.12,.65)],bottom=.28,top=.80)
    for item in list(fig.texts):
        item.remove()  # A single plot does not need a panel letter.
    ax = fig.axes[0]
    ax.set_yticks([1,1e5,1e10,1e15,1e20])
    ax.set_xlabel("Batch normalization / feedback scale")
    ax.set_ylabel("Test cross-entropy")
    ax.legend(frameon=False,fontsize=7,loc="center left",bbox_to_anchor=(1.03,.5),
              handlelength=1.5,handletextpad=.5)
    assert json.loads(measured_artists(fig))[0] == before
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper-figures", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    captured = {}
    previous.m.save = lambda fig, name, scope: captured.update({name: fig})
    previous.theory_and_controls()
    previous.m.paired()
    previous.followups()
    baselines = {name: measured_artists(captured[name]) for name in [*NAMES[1:],REPLICATION]}
    figures = {NAMES[0]: theory(), NAMES[1]: positive(captured[NAMES[1]]),
               NAMES[2]: controls(captured[NAMES[2]]),REPLICATION:compact_replication(captured[REPLICATION])}
    figures[PRACTICAL],practical_record = practical_results()
    figures[STABILITY] = stability_only(captured[STABILITY])
    for index in [1, 3]:
        before = captured[NAMES[0]].axes[index]
        after = figures[NAMES[0]].axes[index]
        for old, new in zip(before.lines, after.lines):
            np.testing.assert_array_equal(old.get_ydata(), new.get_ydata())
    receipt = {"width_inches": 5.5, "figures": {}}
    for name, fig in figures.items():
        if name in baselines:
            assert measured_artists(fig) == baselines[name], "Plotted values changed: " + name
        fig.canvas.draw()
        path = args.output / (name+".pdf")
        fig.savefig(path, metadata={"Author":"", "CreationDate":None, "ModDate":None})
        fig.savefig(args.output/(name+".png"), dpi=200)
        if args.paper_figures:
            args.paper_figures.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, args.paper_figures/path.name)
        receipt["figures"][name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_inches": fig.get_size_inches().tolist(),
            "measurement_artists_unchanged": name in baselines,
            "data_sha256": hashlib.sha256(measured_artists(fig).encode()).hexdigest(),
            "panels": [list(a.get_position().bounds) for a in fig.axes]}
    receipt["figure1_scope"] = "Mixed-residual schematic; unchanged spectral formula and population step means; restored analytic condition-number ratio with explicit labeling."
    receipt["figure5"] = practical_record
    receipt["supplement_followup"] = "Work and width panels moved to main Figure 5; all normalization points and means retained without duplicate plots."
    (args.output/"manifest.json").write_text(json.dumps(receipt, indent=2)+"\n")
    plt.close("all")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
