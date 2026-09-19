"""Publication figures: retained data with explicit replication and scope.

The September 18 generator reproduces the unchanged panels. This pass supplies
recomputed population simulations, a common correlation analysis, paired
follow-up intervals, and a two-panel ImageNet diagnostic.
"""
from pathlib import Path
import copy,importlib.util,json,os,pickle,sys
import numpy as np
import pandas as pd
from scipy.stats import t, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('previous_redraw',ROOT/'scripts/ndfa_revision_20260918/redraw.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.OUT=Path(os.environ.get('NDFA_FIGURE_OUTPUT',ROOT/'drafts/Info-DFA/build/second_review_20260919/figures'))
m.PAPER=Path(os.environ.get('NDFA_PAPER_FIGURES',ROOT/'drafts/Info-DFA/figures'))
DATA=ROOT/'assets/ndfa_revision_20260919'


def followups():
    frame=pd.read_csv(DATA/'followup_endpoints.csv');frame=frame[~frame.development]
    fig,axes=plt.subplots(1,3);m.layout(fig,1,3,height=3.55)
    for ax,phase,key,levels,comparisons,title,xlabel in [
        (axes[0],'work','budget',[60,120],[('ndfa','dfa','a','A − DFA'),('ndfa','bp','bp','A − BP'),('ndfa','fd_dfa',None,'A − FD')],'Longer work','Update work (s)'),
        (axes[1],'width','width',[1024,2048],[('ndfa','dfa','a','A − DFA'),('endfa','dfa','e','E − DFA'),('kndfa','ndfa','k','K − A')],'Retuned error factor','First hidden width')]:
        for i,(a,b,color,label) in enumerate(comparisons):
            color=m.COL[color] if color else '#AA4499';means=[];half=[]
            for x,level in enumerate(levels):
                p=frame[(frame.phase==phase)&(frame[key]==level)].pivot(index='seed',columns='method',values='test_accuracy')
                vals=100*(p[a]-p[b]);assert len(vals)==5 and vals.notna().all()
                means.append(vals.mean());half.append(t.ppf(.975,4)*vals.sem())
                ax.scatter(x+(i-1)*.15+np.linspace(-.025,.025,5),vals,color=color,s=9,alpha=.3,zorder=2)
            ax.errorbar(np.arange(2)+(i-1)*.15,means,yerr=half,fmt='o-',color=color,label=label,ms=3,capsize=2,lw=1.1,zorder=3)
        ax.set_xticks([0,1],[str(x) for x in levels]);ax.set(xlabel=xlabel,ylabel='Accuracy gain (pp)')
        ax.axhline(0,color='.65',lw=.7);ax.set_title(title,fontsize=8,fontweight='bold');ax.margins(x=.16,y=.10)
    for i,(method,color,label) in enumerate([('dfa','dfa','DFA'),('ndfa','a','A'),('endfa','e','E'),('kndfa','k','K')]):
        means=[]
        for x,(bn,fb) in enumerate([(False,.1),(False,1.),(True,.1),(True,1.)]):
            vals=frame[(frame.phase=='stability')&(frame.bn==bn)&(frame.feedback_scale==fb)&(frame.method==method)].test_loss
            assert len(vals)==5;means.append(vals.mean())
            axes[2].scatter(x+(i-1.5)*.1+np.linspace(-.015,.015,5),vals,color=m.COL[color],s=7,alpha=.3)
        axes[2].plot(np.arange(4)+(i-1.5)*.1,means,'o-',color=m.COL[color],label=label,ms=3,lw=1.2)
    axes[2].set(yscale='log',ylabel='Test cross-entropy',ylim=(.6,1e20))
    axes[2].set_yticks([1,1e4,1e8,1e12,1e16,1e20]);axes[2].minorticks_off()
    axes[2].set_xticks(range(4),['Off\n0.1','Off\n1','On\n0.1','On\n1'])
    axes[2].set_xlabel('BN / feedback scale');axes[2].set_title('Predictive stability',fontsize=8,fontweight='bold')
    for i,ax in enumerate(axes):
        x,y,w,h=ax.get_position().bounds;ax.set_position([x,.32,w,.56])
        handles,labels=ax.get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(x+w/2,.015),ncol=2 if i==2 else 1,frameon=False,fontsize=6.5)
    m.save(fig,'ndfa_targeted_followups_20260918','Five paired seeds; individual 95% Student t intervals for differences; fixed split/selected settings. CE uses means and individual endpoints.')


def imagenet():
    data=pd.read_csv(DATA/'imagenet_clean_summary.csv').set_index('tag')
    noisy=pd.read_csv(DATA/'imagenet_noisy_endpoints.csv')
    fig,axes=plt.subplots(1,2);m.layout(fig,1,2,height=3.55)
    a,b=axes;depths=['layer4','l34','l234','all']
    colors={'dfa':m.COL['dfa'],'ndfaDiag':'#44AA99','ndfaFull':'#AA4499'}
    for i,(key,label,marker) in enumerate([('dfa','Raw block-DFA','o'),('ndfaDiag','Diagonal whitening','s'),('ndfaFull','Full ZCA','^')]):
        rows=data.loc[[f'{key}_{d}' for d in depths]];x=np.arange(4)+(i-1)*.11
        a.errorbar(x,rows['mean'],yerr=rows['sem'],fmt=marker+'-',color=colors[key],ms=4,capsize=2,lw=1.3,label=label)
        if key=='ndfaFull':a.plot(x[2:],rows['mean'].iloc[2:],'^',mfc='white',mec=colors[key],ms=4,zorder=4)
    a.axhline(data.loc['bp','mean'],color=m.COL['bp'],ls='--',lw=1,label='BP')
    a.set_xticks(range(4),['layer4','layer3+4','layer2+3+4','all'],rotation=25,ha='right')
    a.set(ylabel='ImageNet-100 top-1 (%)',ylim=(46,87));a.set_title('Substitution depth',fontsize=8,fontweight='bold')
    for j,depth in enumerate(['layer4','all']):
        for i,key in enumerate(['ndfaDiag','ndfaFull']):
            off=(i-.5)*.32;xclean=j*2.7;xn=xclean+1
            delta=data.loc[f'{key}_{depth}','mean']-data.loc[f'dfa_{depth}','mean']
            b.bar(xclean+off,delta,width=.32,facecolor='none',edgecolor=colors[key],lw=1.2,
                  hatch='///' if key=='ndfaFull' and depth=='all' else None)
            p=noisy[noisy.depth==depth].pivot(index='seed',columns='method',values='val_top1')
            d=p[key]-p.dfa;assert len(d)==3
            b.bar(xn+off,d.mean(),width=.32,color=colors[key],yerr=d.sem(),capsize=2,
                  error_kw={'lw':.9,'ecolor':'#333333'})
        b.text(j*2.7+.5,-.19,['layer4','all blocks'][j],transform=b.get_xaxis_transform(),ha='center',fontsize=7)
    b.axhline(0,color='.6',lw=.7);b.set_xticks([0,1,2.7,3.7],['Clean','Noise','Clean','Noise'])
    b.set(ylabel='Whitening − raw DFA (pp)',ylim=(-16.5,11));b.set_title('40% label-noise stress',fontsize=8,fontweight='bold')
    for ax in axes:
        x,y,w,h=ax.get_position().bounds;ax.set_position([x,.37,w,.51])
        ax.grid(axis='y',color='.9',lw=.6);ax.set_axisbelow(True)
    h,l=a.get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.3,.015),ncol=1,frameon=False,fontsize=7)
    fig.legend([Patch(fc=colors['ndfaDiag']),Patch(fc=colors['ndfaFull']),Patch(fc='none',ec='.4'),Patch(fc='.5')],
               ['Diagonal','Full ZCA','Clean','40% noise'],loc='lower center',bbox_to_anchor=(.80,.03),ncol=2,frameon=False,fontsize=7,columnspacing=.7)
    m.save(fig,'iclr_fig5_scaling_boundary','Two ImageNet panels. A mean +/- SEM over three seeds. B clean mean contrasts without intervals; noisy paired mean +/- SEM over three seeds. Different clean/noisy cohorts; full-ZCA deep settings use separate LR.')


def theory_and_controls():
    cache=pickle.loads(Path(os.environ.get('NDFA_FIGURE_CACHE',ROOT/'assets/ndfa_revision_20260918/figure_cache.pkl')).read_bytes())['figures']
    for fig in cache.values():m.recolor_activity(fig)
    linear=pd.read_csv(DATA/'linear_simulation.csv')
    old=cache['iclr_fig_theory_conditioning'].axes[3]
    for line in old.lines:
        if len(line.get_xdata())!=2:continue
        key='activity' if line.get_label()=='nDFA' else 'dfa'
        y=linear[linear.rule==key].groupby('task_high').steps.mean().reindex([False,True]).to_numpy()
        assert np.allclose(line.get_ydata(),y), ('Linear simulation differs from retained figure',line.get_ydata(),y)
        line.set_ydata(y)
    m.theory(cache)
    # legacy_figures reuses the cached control measurements for A/C and derives D
    # at the corrected global-seed level. Rebuild B from one common cell table.
    m.legacy_figures(cache)
    fig=cache['iclr_fig_controls_composite'];ax=fig.axes[1];ax.clear()
    frame=pd.read_csv(DATA/'nuisance_correlation_cells.csv');assert len(frame)==128
    palette=[('nuisance_dominant','Nuisance','#D55E00'),('low_sample_noisy','Low sample','#009E73'),
             ('mixed_context','Mixed','#0072B2'),('task_aligned','Task aligned','#CC79A7')]
    for key,label,color in palette:
        g=frame[frame.condition==key]
        ax.scatter(g.nuisance_energy_ratio,g.gain,s=10,alpha=.45,color=color,label=label)
        ax.scatter(g.nuisance_energy_ratio.mean(),g.gain.mean(),s=27,marker='D',color=color,ec='white',lw=.6)
    rho=spearmanr(frame.nuisance_energy_ratio,frame.gain).statistic
    ax.set(xscale='log',xlabel='Nuisance / task energy',ylabel='A − DFA gain (pp)')
    ax.set_title('Nuisance loading',fontsize=8,fontweight='bold')
    ax.text(.50,.03,rf'$\rho={rho:.2f}$',transform=ax.transAxes,ha='center',va='bottom',fontsize=7)
    ax.legend(frameon=False,fontsize=6,loc='upper left',bbox_to_anchor=(.16,.98),handletextpad=.3,labelspacing=.2)
    ax.grid(color='.92',lw=.6);ax.set_axisbelow(True)
    m.save(fig,'iclr_fig_controls_composite','Controls retained; B uses the common 128-cell rank-selected analysis (rho=.623327); D uses five global-seed means.')


def main():
    m.main()
    theory_and_controls();followups();imagenet()
    (m.OUT/'manifest.json').write_text(json.dumps(m.RECORDS,indent=2)+'\n')
    print('Final publication figures:',len(m.RECORDS))


if __name__=='__main__':main()
