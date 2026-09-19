"""Publication figures from retained measurements and corrected summary units."""
import os
import copy
import importlib.util
import json
import pickle
from pathlib import Path
import re
import sys
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.collections import PolyCollection
from matplotlib.patches import Ellipse, FancyArrowPatch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=Path(os.environ.get('NDFA_FIGURE_OUTPUT',ROOT/'drafts/Info-DFA/build/comments_revision_20260918/figures'))
PAPER=Path(os.environ.get('NDFA_PAPER_FIGURES',ROOT/'drafts/Info-DFA/figures'))
COL={'bp':'#0072B2','dfa':'#7F7F7F','a':'#009E73','e':'#D55E00','k':'#6A3D9A'}
RECORDS={}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
                    'xtick.labelsize':7.5,'ytick.labelsize':7.5,'pdf.fonttype':42,
                    'axes.spines.top':False,'axes.spines.right':False})


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def layout(fig,rows,cols,*,height=3.7):
    fig.set_layout_engine(None);fig.set_size_inches(5.5,height)
    for t in list(fig.texts):t.remove()
    for legend in list(fig.legends):legend.remove()
    width=(.86-.12*(cols-1))/cols
    h=(.70-.20*(rows-1))/rows
    for i,ax in enumerate(fig.axes):
        row,col=divmod(i,cols)
        x=.12+col*(width+.12);y=.18+(rows-1-row)*(h+.20)
        ax.set_position([x,y,width,h])
        for t in list(ax.texts):
            if re.fullmatch('[A-Z]',t.get_text().strip()):t.remove()
        for title in [ax.title,ax._left_title,ax._right_title]:
            title.set_text(re.sub(r'^[A-Z]\s{1,2}','',title.get_text()))
            title.set_fontsize(8);title.set_fontweight('bold')
        ax.tick_params(labelsize=7.5,pad=2)
        ax.xaxis.label.set_size(8);ax.yaxis.label.set_size(8)
        if ax.get_legend():
            for t in ax.get_legend().get_texts():t.set_fontsize(7)
        fig.text(x-.09,y+h+.06,chr(65+i),fontsize=10,fontweight='bold',va='bottom')


def save(fig,name,scope):
    OUT.mkdir(parents=True,exist_ok=True);PAPER.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw()
    data=[]
    for ax in fig.axes:
        data.append({'bounds':list(ax.get_position().bounds),
                     'lines':[{'x':np.asarray(v.get_xdata(),float).tolist(),'y':np.asarray(v.get_ydata(),float).tolist()} for v in ax.lines]})
    for target in (OUT,PAPER):
        fig.savefig(target/(name+'.pdf'),metadata={'Author':'','CreationDate':None,'ModDate':None})
    fig.savefig(OUT/(name+'.png'),dpi=180)
    RECORDS[name]={'scope':scope,'axes':data,'sha256':hashlib.sha256((PAPER/(name+'.pdf')).read_bytes()).hexdigest()}
    plt.close(fig)


def theory(cache):
    old=cache['iclr_fig_theory_conditioning']
    fig,axes=plt.subplots(2,2)
    layout(fig,2,2,height=3.5)
    a,b,c,d=axes.ravel()
    a.set_xlim(-.12,1.4);a.set_ylim(-.12,1.35);a.axis('off')
    a.add_patch(Ellipse((.55,.45),1.4,.42,fc='#DAEAE1',ec='none'))
    for end,color,label,xy in [((.30,.98),'#222222','Mixed residual',(.33,1.02)),
                               ((1.25,.32),COL['dfa'],'Raw DFA',(.94,.05)),
                               ((.31,.91),COL['a'],'Conditioned',(.38,.77))]:
        a.add_patch(FancyArrowPatch((0,0),end,arrowstyle='-|>',mutation_scale=10,lw=1.6,color=color))
        a.text(*xy,label,color=color,fontsize=7.5)
    a.text(.58,-.10,'Nuisance direction',ha='center',fontsize=7.5)
    a.text(-.10,.65,'Task',rotation=90,ha='center',fontsize=7.5)
    a.set_title('A mixed update',fontsize=8,fontweight='bold')
    lam=np.geomspace(1,70,12);gain=lam/(lam+.001)
    b.plot(range(1,13),lam/lam.max(),'o--',color=COL['dfa'],ms=3,label='BP / DFA')
    b.plot(range(1,13),gain/gain.max(),'o-',color=COL['a'],ms=3,label='Activity')
    b.set(xlabel='Eigendirection',ylabel='Relative weight',ylim=(-.04,1.12))
    b.legend(frameon=False,fontsize=7,loc='center left');b.set_title('Spectral weighting',fontsize=8,fontweight='bold')
    c.axis('off');c.set_title('Conditioning the local update',fontsize=8,fontweight='bold')
    for y,label,formula,color in [(.77,'Activity',r'$G P_A$',COL['a']),(.45,'Error',r'$P_E G$',COL['e']),(.13,'Both',r'$P_E G P_A$',COL['k'])]:
        c.text(.02,y,label,color=color,fontsize=8,va='center');c.text(.58,y,formula,fontsize=12,va='center')
    # The retained simulation means are the exact plotted source of the old D.
    source=old.axes[3]
    for line in source.lines:
        if len(line.get_xdata())==2:
            d.plot(line.get_xdata(),line.get_ydata(),line.get_marker(),color=COL['a'] if line.get_label()=='nDFA' else COL['dfa'],ms=5,label=line.get_label())
    d.set_yscale('log');d.set_ylim(source.get_ylim());d.set_xticks([0,1],['Nuisance high','Task high'])
    d.set_ylabel('Steps to target');d.set_title('Task orientation',fontsize=8,fontweight='bold')
    d.legend(frameon=False,fontsize=7,loc='upper right')
    save(fig,'iclr_fig_theory_conditioning','Retained spectral/step measurements; explicit mixed residual and A/E/K schematic replace bound panel.')


def recolor_activity(fig):
    from matplotlib.colors import to_rgba
    old=np.asarray(to_rgba('#0072B2'))
    for artist in fig.findobj():
        for getter,setter in [('get_color','set_color'),('get_facecolor','set_facecolor'),('get_edgecolor','set_edgecolor')]:
            if not hasattr(artist,getter) or not hasattr(artist,setter):continue
            try:
                color=getattr(artist,getter)()
                a=np.asarray(color) if not isinstance(color,str) else np.asarray(to_rgba(color))
                if a.ndim==1 and len(a) in (3,4) and np.allclose(a[:3],old[:3]):
                    getattr(artist,setter)(COL['a'])
                elif a.ndim==2 and a.shape[1]==4:
                    mask=np.all(np.isclose(a[:,:3],old[:3]),axis=1)
                    if mask.any():
                        a[mask,:3]=to_rgba(COL['a'])[:3];getattr(artist,setter)(a)
            except (ValueError,TypeError):pass


def legacy_figures(cache):
    fig=cache['iclr_fig1_rule_and_positive_regimes'];layout(fig,2,2,height=3.7)
    for ax in fig.axes[:3]:
        for item in list(ax.collections):
            if isinstance(item,PolyCollection):item.remove()
    for ax,title in zip(fig.axes,['Nuisance-dominant · 14 epochs','Low-sample/noisy · 14 epochs','Mixed-context · 14 epochs','Activity effect · 100 epochs']):
        ax.set_title(title,fontsize=8,fontweight='bold')
    handles,labels=fig.axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False,fontsize=7)
    save(fig,'iclr_fig1_rule_and_positive_regimes','Retained curves and controls; explicit horizons and larger panels.')
    fig=cache['iclr_fig_controls_composite'];layout(fig,2,2,height=3.7)
    # Recompute the alignment band across model seeds after feedback averaging.
    ax=fig.axes[3];ax.clear()
    frame=pd.read_csv(ROOT/'results/infodfa_alignment_dynamics_v1/alignment_dynamics.csv')
    for key,label,color in [('dfa_random','DFA',COL['dfa']),('ndfa_random','A',COL['a'])]:
        group=frame[(frame.cell=='nuisance_hard')&(frame.method==key)]
        group=group.groupby(['seed','epoch']).weight_align_mean.mean().groupby('epoch').agg(['mean','sem'])
        x=group.index.to_numpy();y=group['mean'].to_numpy();e=group['sem'].to_numpy()
        ax.plot(x,y,color=color,label=label);ax.fill_between(x,y-e,y+e,color=color,alpha=.15,lw=0)
    ax.axhline(0,color='.6',lw=.6);ax.set(xlabel='Epoch',ylabel='Weight alignment');ax.set_title('Alignment acquisition',fontsize=8,fontweight='bold');ax.legend(frameon=False,fontsize=7)
    fig.axes[0].set_xticks(fig.axes[0].get_xticks(),['Nuis.','Low-N','Mixed'],rotation=0)
    for t in list(fig.axes[1].texts):
        if 'Spearman' in t.get_text():t.remove()
    for ax in fig.axes[:3]:
        ax.tick_params(labelsize=7)
        for t in ax.texts:t.set_fontsize(6.5)
    save(fig,'iclr_fig_controls_composite','Alignment band uses five global seeds; retained other controls; larger panels.')
    fig=cache['infodfa_feedback_variance'];layout(fig,1,2,height=3.2)
    ax=fig.axes[1];ax.clear()
    cells=pd.read_csv(ROOT/'results/infodfa_feedback_variance_v1/feedback_variance_cells.csv')
    for method,label,color in [('dfa_random','DFA',COL['dfa']),('ndfa_random','A',COL['a'])]:
        sub=cells[(cells.method==method)&(cells.sd_fb>0)&~cells.at_floor]
        x=100*sub.mean_acc;y=100*sub.sd_fb
        ax.scatter(x,y,s=8,color=color,alpha=.5)
        slope,intercept=np.polyfit(x,np.log10(y),1);xx=np.linspace(x.min(),x.max(),50)
        ax.plot(xx,10**(intercept+slope*xx),color=color,label=label)
    ax.set(yscale='log',xlabel='Mean test accuracy (%)',ylabel='Feedback-seed SD (pp)')
    ax.set_title('Dependence on accuracy',fontsize=8,fontweight='bold');ax.legend(frameon=False,fontsize=7)
    fig.axes[0].set_title('Feedback variability',fontsize=8,fontweight='bold')
    handles,labels=fig.axes[0].get_legend_handles_labels();fig.axes[0].get_legend().remove()
    short=['Nuisance','Low sample','Mixed','Task aligned','Fashion-MNIST','CIFAR-10','Mixer','DFA at chance']
    fig.legend(handles,short,loc='lower center',ncol=4,frameon=False,fontsize=6.5,columnspacing=.9)
    for axis in fig.axes:
        x,y,w,h=axis.get_position().bounds;axis.set_position([x,.27,w,.61])
    fig.axes[0].set(xlabel='DFA feedback SD (pp)',ylabel='Activity feedback SD (pp)')
    fig.axes[1].set_ylabel('Feedback SD (pp)')
    save(fig,'infodfa_feedback_variance','Obsolete K trace removed; raw/A cell measurements retained.')


def paired():
    from analysis import make_error_kndfa_replication_figure as m
    frames=m.load_seed_means();fig,axes=plt.subplots(1,3);layout(fig,1,3,height=2.55)
    for ax,method,reference,color,title in [(axes[0],'endfa','dfa',COL['e'],'E − DFA'),(axes[1],'kndfa','ndfa',COL['k'],'K − A')]:
        differences=m.paired_deltas(frames,method,reference)
        for i,(name,values) in enumerate(differences.items()):
            y=values.to_numpy();ax.scatter(i+np.linspace(-.09,.09,len(y)),y,s=12,color=color,alpha=.55)
            mean=y.mean();sem=y.std(ddof=1)/np.sqrt(len(y))
            ax.errorbar(i,mean,yerr=sem,fmt='o',color=color,ms=4,capsize=2)
            ax.annotate(f'{mean:+.2f}',(i,max(y)),xytext=(5 if i==0 else 0,6),textcoords='offset points',ha='center',fontsize=7)
        ax.axhline(0,color='.6',ls=':',lw=.7);ax.set_xticks(range(3),['MNIST\ntanh','Fashion\ntanh','MNIST\nReLU'])
        ax.set_ylabel('Accuracy gain (pp)');ax.set_title(title,fontsize=8,fontweight='bold');ax.margins(y=.22)
    wide=frames['ReLU MNIST'].pivot(index='seed',columns='method',values='test_loss')
    for i,(key,color) in enumerate(zip(['dfa','ndfa','endfa','kndfa'],[COL['dfa'],COL['a'],COL['e'],COL['k']])):
        y=wide[key].to_numpy();axes[2].scatter(i+np.linspace(-.1,.1,len(y)),y,s=12,color=color,alpha=.65)
        axes[2].plot(i,y.mean(),'_',color=color,ms=12,mew=1.5)
    axes[2].set(yscale='log',ylabel='Test cross-entropy');axes[2].set_xticks(range(4),['DFA','A','E','K']);axes[2].set_title('Predictive loss',fontsize=8,fontweight='bold')
    save(fig,'iclr_fig_error_kndfa_replication','Original five/five/eight seed cohorts; mean gain annotations and predictive-loss panel.')


def spatial(cache,audit):
    fig=cache['iclr_supp_spatial_kron'];layout(fig,1,2,height=2.7)
    ax=fig.axes[0];ax.clear()
    for dataset,color,marker in [('cifar10','#882255','o'),('cifar100','#44AA99','^')]:
        rows=[r for r in audit['spatial'] if r['dataset']==dataset]
        for r in rows:
            ax.scatter(r['amplitude']+np.linspace(-.035,.035,5),r['initialization_seed_means_pp'],s=10,color=color,alpha=.4)
        ax.plot([r['amplitude'] for r in rows],[r['mean_difference_pp'] for r in rows],marker+'-',color=color,label=dataset.upper().replace('CIFAR','CIFAR-'))
    ax.axhline(0,color='.6',ls=':',lw=.7);ax.set(xlabel='Nuisance amplitude',ylabel='Spatial − channel gain (pp)');ax.set_title('Kernel-patch effect',fontsize=8,fontweight='bold');ax.legend(frameon=False,fontsize=7)
    fig.axes[1].legend(frameon=False,fontsize=7,loc='upper left');fig.axes[1].set_title('Damping dependence',fontsize=8,fontweight='bold')
    save(fig,'iclr_supp_spatial_kron','Complete crossed-design means and five initialization means; invalid independent-pair intervals removed.')
    # The old composite carries the same invalid interval; replace only that panel.
    fig=cache['iclr_fig5_scaling_boundary'];layout(fig,1,3,height=2.8)
    ax=fig.axes[1];ax.clear();rows=[r for r in audit['spatial'] if r['dataset']=='cifar10']
    ax.plot([r['amplitude'] for r in rows],[r['mean_difference_pp'] for r in rows],'o-',color='#882255')
    for r in rows:ax.scatter(np.repeat(r['amplitude'],5),r['initialization_seed_means_pp'],s=8,color='#882255',alpha=.4)
    ax.axhline(0,color='.6',lw=.7);ax.set(xlabel='Nuisance amplitude',ylabel='Spatial − channel (pp)');ax.set_title('Kernel-patch effect',fontsize=8,fontweight='bold')
    fig.axes[2].set_title('Label-noise stress',fontsize=8,fontweight='bold')
    fig.axes[2].set_title('',loc='left')
    save(fig,'iclr_fig5_scaling_boundary','ImageNet composite retained in supplement; spatial uncertainty shown as seed points.')


def practical():
    m=module(ROOT/'scripts/visual_revision_20260915/redesign.py','review_practical')
    m.COLORS.update(bp=COL['bp'],ndfa=COL['a'],kndfa=COL['k'])
    def output(fig,name,letters,record):
        for letter,x,y in letters:fig.text(x,y,letter,fontsize=10,fontweight='bold',va='bottom')
        save(fig,name,'Accepted matched-work and factor-transfer contrasts; unified colors.')
    m.save_new=output;m.final_results()


def mechanism():
    m=module(ROOT/'analysis/plot_ndfa_factor_mechanism.py','review_mechanism')
    audit,curves,endpoints,cosine=m.load(ROOT/'docs/research/ndfa_factor_mechanism_20260915.json',ROOT/'results/ndfa_factor_mechanism_20260915')
    m.COLORS.update(raw=COL['dfa'],activity=COL['a'],error=COL['e'],kronecker=COL['k'],bp=COL['bp'])
    fig=plt.figure(figsize=(5.5,4.2));positions=[.14,.42,.70]
    for i,method in enumerate(['activity','error','kronecker']):
        ax=fig.add_axes([positions[i],.595,.21,.275]);v=np.array([[cosine[(a,e,method)] for e in m.ORIENTATIONS] for a in m.ORIENTATIONS])
        ax.imshow(v,vmin=0,vmax=1,cmap='RdBu')
        for r in range(3):
            for c in range(3):ax.text(c,r,f'{v[r,c]:.2f}',ha='center',va='center',fontsize=7,color='white' if v[r,c]<.25 or v[r,c]>.85 else '#222222')
        ax.set_xticks(range(3),['Iso.','Low','High']);ax.set_yticks(range(3),['Iso.','Low','High'] if i==0 else [])
        ax.set_xlabel('Error orientation');ax.set_title(['Activity','Error','Both'][i],fontsize=8,fontweight='bold')
        if i==0:ax.set_ylabel('Activity orientation')
        fig.text(positions[i]-.065,.925,chr(65+i),fontsize=10,fontweight='bold')
    lower=[]
    for i,orientation in enumerate(m.ORIENTATIONS):
        ax=fig.add_axes([positions[i],.17,.21,.25]);m.trajectory(ax,curves,endpoints,orientation,orientation,title=['Isotropic','Signal low','Signal high'][i],ylabel=i==0)
        ax.set_title('',loc='left');ax.set_title(['Isotropic','Signal low','Signal high'][i],fontsize=8,fontweight='bold');ax.set_xlabel('Update');ax.set_xticks([0,50,100]);ax.tick_params(labelsize=7)
        fig.text(positions[i]-.065,.475,chr(68+i),fontsize=10,fontweight='bold');lower.append(ax)
    handles,labels=lower[0].get_legend_handles_labels();fig.legend(handles,['DFA','A','E','K','BP','DFA stationary loss'],loc='lower center',ncol=3,frameon=False,fontsize=7)
    save(fig,'ndfa_factor_mechanism_20260915','Full declared orientation slice; aligned rows/columns and common method colors; interpretation in caption.')


def factor_confirmations():
    from analysis import make_error_kndfa_replication_figure as m
    frames=m.load_seed_means()
    for label,directory,name,ridges in [
        ('tanh MNIST', 'dfa_stall_threefactor_analysis_v1','iclr_fig_threefactor_conditioning',(.3,10)),
        ('tanh Fashion','dfa_stall_fashion_threefactor_analysis_v1','iclr_fig_fashion_threefactor_conditioning',(.03,30)),
        ('ReLU MNIST','dfa_relu_mnist_threefactor_analysis_v1','iclr_fig_relu_threefactor_conditioning',(3,.1))]:
        selected=pd.read_csv(ROOT/'results'/directory/'damping_selection.csv')
        frame=frames[label]
        methods=['dfa','ndfa','endfa','kndfa'];colors=[COL['dfa'],COL['a'],COL['e'],COL['k']];labels=['DFA','A','E','K']
        if label=='tanh Fashion':
            extra=pd.read_csv(ROOT/'results'/directory/'confirmation_seed_means.csv')
            extra=extra[(extra.method=='kndfa_bp')&extra.seed.isin(range(70,75))]
            assert len(extra)==5 and extra.n_feedback_seeds.eq(3).all()
            frame=pd.concat([frame,extra]);methods+=['kndfa_bp'];colors+=['#CC79A7'];labels+=['BP-E']
        fig,axes=plt.subplots(1,3);layout(fig,1,3,height=2.55)
        for i,(side,color) in enumerate([('activity',COL['a']),('error',COL['e'])]):
            g=selected[selected.side==side].sort_values('damping')
            axes[i].errorbar(g.damping,100*g.validation_acc_mean,yerr=100*g.validation_acc_sem,color=color,marker='o',ms=3,capsize=2)
            axes[i].axvline(ridges[i],color='.4',ls=':',lw=.7);axes[i].set_xscale('log')
            axes[i].set_xlabel(r'$\lambda_A$' if i==0 else r'$\lambda_E$');axes[i].set_title(side.capitalize()+' damping',fontsize=8,fontweight='bold')
        axes[0].set_ylabel('Validation accuracy (%)')
        wide=frame.pivot(index='seed',columns='method',values='test_acc')[methods]*100
        for _,row in wide.iterrows():axes[2].plot(range(len(methods)),row,color='.75',lw=.6,marker='o',ms=2)
        for i,color in enumerate(colors):axes[2].errorbar(i,wide.iloc[:,i].mean(),yerr=wide.iloc[:,i].sem(),fmt='o',ms=4,color=color,capsize=2)
        axes[2].set_xticks(range(len(methods)),labels,rotation=0);axes[2].set_ylabel('Test accuracy (%)');axes[2].set_title('Confirmation',fontsize=8,fontweight='bold')
        save(fig,name,'Original declared confirmation cohort and independent development sweeps; enlarged aligned panel letters.')


def followups():
    directory=ROOT/'results/ndfa_revision_followups_audit_20260918'
    if not (directory/'audit.json').exists() or not json.loads((directory/'audit.json').read_text())['complete']:return
    frame=pd.read_csv(directory/'endpoints.csv');frame=frame[~frame.development]
    fig,axes=plt.subplots(1,3);layout(fig,1,3,height=3.45)
    for i,(a,b,color,label) in enumerate([('ndfa','dfa',COL['a'],'A − DFA'),('ndfa','bp',COL['bp'],'A − BP'),('ndfa','fd_dfa','#AA4499','A − FD')]):
        means=[]
        for x,budget in enumerate([60,120]):
            pivot=frame[(frame.phase=='work')&(frame.budget==budget)].pivot(index='seed',columns='method',values='test_accuracy')
            vals=100*(pivot[a]-pivot[b]);means.append(vals.mean());axes[0].scatter(x+(i-1)*.10+np.linspace(-.02,.02,5),vals,color=color,s=7,alpha=.35)
        axes[0].plot(np.arange(2)+(i-1)*.1,means,'o-',color=color,label=label,ms=3)
    axes[0].set_xticks([0,1],['60','120']);axes[0].set(xlabel='Update work (s)',ylabel='Accuracy gain (pp)');axes[0].axhline(0,color='.7',lw=.7);axes[0].set_title('Longer work',fontsize=8,fontweight='bold');axes[0].legend(frameon=False,fontsize=6.5)
    for i,(a,b,color,label) in enumerate([('ndfa','dfa',COL['a'],'A − DFA'),('endfa','dfa',COL['e'],'E − DFA'),('kndfa','ndfa',COL['k'],'K − A')]):
        means=[]
        for x,width in enumerate([1024,2048]):
            pivot=frame[(frame.phase=='width')&(frame.width==width)].pivot(index='seed',columns='method',values='test_accuracy');vals=100*(pivot[a]-pivot[b]);means.append(vals.mean());axes[1].scatter(x+(i-1)*.10+np.linspace(-.02,.02,5),vals,color=color,s=7,alpha=.35)
        axes[1].plot(np.arange(2)+(i-1)*.1,means,'o-',color=color,label=label,ms=3)
    axes[1].set_xticks([0,1],['1024','2048']);axes[1].set(xlabel='First hidden width',ylabel='Accuracy gain (pp)');axes[1].axhline(0,color='.7',lw=.7);axes[1].set_title('Retuned error factor',fontsize=8,fontweight='bold');axes[1].legend(frameon=False,fontsize=6.5)
    for i,(method,color,label) in enumerate([('dfa',COL['dfa'],'DFA'),('ndfa',COL['a'],'A'),('endfa',COL['e'],'E'),('kndfa',COL['k'],'K')]):
        means=[]
        for x,(bn,fb) in enumerate([(False,.1),(False,1.),(True,.1),(True,1.)]):
            vals=frame[(frame.phase=='stability')&(frame.bn==bn)&(frame.feedback_scale==fb)&(frame.method==method)].test_loss
            assert len(vals)==5;means.append(vals.mean());axes[2].scatter(x+(i-1.5)*.1+np.linspace(-.015,.015,5),vals,color=color,s=6,alpha=.35)
        axes[2].plot(np.arange(4)+(i-1.5)*.1,means,'o-',color=color,label=label,ms=3)
    axes[2].set(yscale='log',ylabel='Test cross-entropy');axes[2].set_xticks(range(4),['Off\n0.1','Off\n1','On\n0.1','On\n1']);axes[2].set_xlabel('BN / feedback scale');axes[2].set_title('Predictive stability',fontsize=8,fontweight='bold');axes[2].legend(frameon=False,fontsize=6.5,ncol=2,loc='upper right')
    for i,ax in enumerate(axes):
        x,y,w,h=ax.get_position().bounds;ax.set_position([x,.32,w,.56])
        handles,labels=ax.get_legend_handles_labels();ax.get_legend().remove()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(x+w/2,.012),ncol=2 if i==2 else 1,frameon=False,fontsize=6.5)
    save(fig,'ndfa_targeted_followups_20260918','All planned follow-up conditions; all seed differences or absolute CE endpoints; no selection on test.')


def main():
    cachepath=Path(os.environ.get('NDFA_FIGURE_CACHE',ROOT/'assets/ndfa_revision_20260918/figure_cache.pkl'))
    cache=pickle.loads(cachepath.read_bytes())['figures']
    for fig in cache.values():recolor_activity(fig)
    audit=json.loads((ROOT/'results/ndfa_revision_saved_audit_20260918/audit.json').read_text())
    theory(cache);legacy_figures(cache);paired();spatial(cache,audit);practical();mechanism();factor_confirmations();followups()
    (OUT/'manifest.json').write_text(json.dumps(RECORDS,indent=2)+'\n')
    print('Redrawn',len(RECORDS),'figures into',OUT)


if __name__=='__main__':main()
