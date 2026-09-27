"""Label and readability corrections using released historical inputs only.

No training, selection, pooling, or changes to statistical estimators.
Figure 5 is generated separately by the manuscript's confirmation_material.py.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import pickle
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
DATA=ROOT/'assets/ndfa_arxiv_review_20260926'
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
layout=module(ROOT/'scripts/ndfa_figure_layout_20260919/redraw.py','review_layout')
base=layout.previous.m
COL=base.COL

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'build/arxiv_review_figures_20260926')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    for name,item in json.loads((DATA/'MANIFEST.json').read_text())['files'].items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==item['sha256'],name
    records={}
    def save(fig,name,note,before=None):
        after=layout.measured_artists(fig)
        if before is not None:assert after==before,(name,'plotted measurements changed')
        fig.canvas.draw()
        fig.savefig(args.paper/'figures'/f'{name}.pdf',metadata={'Author':'','CreationDate':None,'ModDate':None})
        fig.savefig(args.output_dir/f'{name}.png',dpi=200)
        records[name]={'note':note,'measurements_sha256':hashlib.sha256(after.encode()).hexdigest(),
                       'unchanged_artist_values_checked':before is not None}
        plt.close(fig)
    # The cache is a versioned, trusted project asset, not an external pickle.
    with (ROOT/'assets/ndfa_revision_20260918/figure_cache.pkl').open('rb') as f:cache=pickle.load(f)["figures"]
    fig=layout.theory();before=layout.measured_artists(fig)
    a=fig.axes[0];a.lines[0].set_zorder(4);a.lines[0].set_linewidth(1.0)
    a.text(.48,1.25,'Residual',fontsize=6.2,ha='left',color='#444444')
    d=fig.axes[3]
    for line in d.lines:
        if line.get_label()=='DFA':line.set_label('Aligned DFA')
    layout.small_legend(d,loc='upper right')
    save(fig,'iclr_fig_theory_conditioning','Residual visibility; explicitly aligned linear DFA.',before)

    fig=cache['iclr_fig1_rule_and_positive_regimes']
    for ax in fig.axes[:3]:
        for item in list(ax.collections):
            if isinstance(item,PolyCollection):item.remove()
    base.recolor_activity(fig)
    layout.positive(fig);before=layout.measured_artists(fig)
    for ax in fig.axes[:3]:
        for line in ax.lines:
            if len(line.get_xdata())==1 and line.get_marker()=='*':
                line.set_markersize(7);line.set_markeredgecolor('white');line.set_markeredgewidth(.45);line.set_zorder(8)
    save(fig,'iclr_fig1_rule_and_positive_regimes','Larger tuned-BP stars; all curve and endpoint values retained.',before)

    from analysis import make_error_kndfa_replication_figure as cohorts
    paths={'tanh MNIST':'dfa_stall','tanh Fashion':'dfa_stall_fashion','ReLU MNIST':'dfa_relu_mnist'}
    frames={k:cohorts.original_cohort(pd.read_csv(ROOT/'assets/ndfa_strengthening_20260925'/f'{v}_threefactor_original_seed_means.csv'),k) for k,v in paths.items()}
    original_loader=cohorts.load_seed_means;original_save=base.save
    captured=[]
    cohorts.load_seed_means=lambda:frames
    base.save=lambda fig,*args:captured.append(fig)
    try:base.paired()
    finally:cohorts.load_seed_means=original_loader;base.save=original_save
    fig=captured[0];layout.compact_replication(fig);before=layout.measured_artists(fig)
    for ax,method,reference in [(fig.axes[0],'endfa','dfa'),(fig.axes[1],'kndfa','ndfa')]:
        for i,v in enumerate(cohorts.paired_deltas(frames,method,reference).values()):
            ax.annotate(f'{v.mean():+.2f}',(i,v.max()),xytext=(0,5),textcoords='offset points',ha='center',fontsize=6.5)
    save(fig,'iclr_fig_error_kndfa_replication','Mean annotations for original five/five/eight-seed cohorts.',before)

    fig=cache['infodfa_feedback_variance'];base.layout(fig,1,2,height=3.2)
    ax=fig.axes[1];ax.clear();cells=pd.read_csv(DATA/'feedback_variance_cells.csv')
    for method,label,color in [('dfa_random','DFA',COL['dfa']),('ndfa_random','A-nDFA',COL['a'])]:
        sub=cells[(cells.method==method)&(cells.sd_fb>0)&~cells.at_floor]
        x=100*sub.mean_acc;y=100*sub.sd_fb
        ax.scatter(x,y,s=8,color=color,alpha=.5)
        slope,intercept=np.polyfit(x,np.log10(y),1);xx=np.linspace(x.min(),x.max(),50)
        ax.plot(xx,10**(intercept+slope*xx),color=color,label=label)
    ax.set(yscale='log',xlabel='Mean test accuracy (%)',ylabel='Feedback/order SD (pp)')
    ax.set_title('Dependence on accuracy',fontsize=8,fontweight='bold');ax.legend(frameon=False,fontsize=7)
    fig.axes[0].set_title('Feedback/order variability',fontsize=8,fontweight='bold')
    handles,labels=fig.axes[0].get_legend_handles_labels();fig.axes[0].get_legend().remove()
    fig.legend(handles,['Nuisance','Low sample','Mixed','Task aligned','Fashion-MNIST','CIFAR-10','MLP-Mixer','DFA at chance'],loc='lower center',ncol=4,frameon=False,fontsize=6.5,columnspacing=.9)
    for axis in fig.axes:
        x,y,w,h=axis.get_position().bounds;axis.set_position([x,.27,w,.61])
    fig.axes[0].set(xlabel='DFA feedback/order SD (pp)',ylabel='A-nDFA feedback/order SD (pp)')
    save(fig,'infodfa_feedback_variance','Compound feedback/order randomization; unchanged cell subset and fitted trends.')

    selected=pd.read_csv(DATA/'damping_selection.csv');fig,axes=plt.subplots(1,3);base.layout(fig,1,3,height=2.55)
    for i,(side,color,ridge) in enumerate([('activity',COL['a'],.3),('error',COL['e'],10)]):
        g=selected[selected.side==side].sort_values('damping')
        axes[i].errorbar(g.damping,100*g.validation_acc_mean,yerr=100*g.validation_acc_sem,color=color,marker='o',ms=3,capsize=2)
        axes[i].axvline(ridge,color='.4',ls=':',lw=.7);axes[i].set_xscale('log')
        axes[i].set_xlabel(r'$\lambda_A$' if i==0 else r'$\lambda_E$');axes[i].set_title(side.capitalize()+' damping',fontsize=8,fontweight='bold')
    axes[0].set_ylabel('Validation accuracy (%)')
    wide=frames['tanh MNIST'].pivot(index='seed',columns='method',values='test_acc')[['dfa','ndfa','endfa','kndfa']]*100
    for _,row in wide.iterrows():axes[2].plot(range(4),row,color='.75',lw=.6,marker='o',ms=2)
    for i,color in enumerate([COL['dfa'],COL['a'],COL['e'],COL['k']]):axes[2].errorbar(i,wide.iloc[:,i].mean(),yerr=wide.iloc[:,i].sem(),fmt='o',ms=4,color=color,capsize=2)
    axes[2].set_xticks(range(4),['DFA','A','E','K']);axes[2].set_ylabel('Test accuracy (%)');axes[2].set_title('Historical MNIST',fontsize=8,fontweight='bold')
    save(fig,'iclr_fig_threefactor_conditioning','Original exposed MNIST seed cohort, not a test-blind confirmation.')

    fig,details=layout.practical_results(DATA/'summary.json');before=layout.measured_artists(fig)
    fig.axes[3].set_xlabel('First hidden width',fontsize=7)
    save(fig,'ndfa_final_test_summary_20260915','Width label corrected; all 21 historical contrasts retained.',before)
    records['historical_practical_checks']=details
    mode=module(Path(__file__).with_name('mode_timing_plot.py'),'review_mode_timing')
    save(mode.fig,'mode_timing_validation','Expected excess-risk labels; all five archived numerical output tables retained.')
    (args.output_dir/'receipt.json').write_text(json.dumps(records,indent=2)+'\n')
    print('Redrew seven historical figures from released inputs; no training or selection.')
if __name__=='__main__':main()
