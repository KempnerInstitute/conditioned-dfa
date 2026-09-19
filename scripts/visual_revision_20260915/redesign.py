"""Presentation-only revision from frozen PDFs and accepted final-test records.

Existing plot paths are preserved; only panel headings/letters move. New plots
read the complete accepted comparisons, without training or model selection.
"""
from pathlib import Path
import hashlib
import json
import re
import copy
import argparse
import shutil

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'drafts/Info-DFA/build/visual_revision_20260915'
WORK = BASE / 'workspace'
PREVIEW = BASE / 'figure_review'
SOURCE = ROOT / 'drafts/Info-DFA/build/iclr_completion_20260915/release_02/source_snapshot'
SHA = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
REPORT = {'figures': []}
SUMMARY = ROOT/'docs/research/ndfa_final_test_20260915/summary.json'
COLORS = {'bp':'#222222', 'dfa':'#7F7F7F', 'ndfa':'#0072B2', 'endfa':'#D55E00', 'kndfa':'#009E73', 'fd_dfa':'#9467BD'}
FONT = font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans', weight='bold'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.labelsize':7.5,
                     'xtick.labelsize':7.2,'ytick.labelsize':7.2,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})


def spans(page):
    return [s for b in page.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans']]


def geometry_signature(page):
    """Retain every vector path, stroke, fill, and dash; ignore stream ordering."""
    keys = ('items','type','color','fill','width','dashes','lineCap','lineJoin','closePath','fill_opacity','stroke_opacity')
    return [{k: d.get(k) for k in keys} for d in page.get_drawings()]


def existing_figures():
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    for name, digest in manifest['file_sha256'].items():
        if not name.endswith('.pdf'): continue
        source=SOURCE/name
        assert SHA(source)==digest
        doc=fitz.open(source); page=doc[0]
        labels=[s for s in spans(page) if re.match(r'^[A-H](?:\s{2}|$)',s['text']) and
                'Bold' in s['font'] and s['size']>=7.9]
        if not labels:
            shutil.copyfile(source,WORK/name)
            page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(PREVIEW/(Path(name).stem+'.png'))
            REPORT['figures'].append({'name':name,'changed':False,'reason':'Single-panel figure; no panel letters.'})
            continue
        labels.sort(key=lambda s:s['text'][0])
        assert ''.join(s['text'][0] for s in labels)==''.join(chr(65+i) for i in range(len(labels))), name
        original_paths=geometry_signature(page)
        # Remove the old heading text only: never redact vector data or images.
        for s in labels:
            page.add_redact_annot(fitz.Rect(s['bbox']), fill=False, cross_out=False)
        page.apply_redactions(images=0, graphics=0, text=0)
        assert geometry_signature(page)==original_paths, name+' vector geometry changed'
        page.insert_font(fontname='hebo')
        x0=min(s['bbox'][0] for s in labels)
        columns=[]
        for value in sorted(s['bbox'][0] for s in labels):
            if not columns or abs(value-columns[-1])>1:columns.append(value)
        placements=[]
        for s in labels:
            letter=s['text'][0]
            title=s['text'][1:].strip()
            title={'Optimized accuracy':'Accuracy','Optimized loss':'Cross-entropy',
                   'Task: low variance':'Low-variance task','Task: high variance':'High-variance task'}.get(title,title)
            title=title.replace('−','-')
            if title:
                page.insert_text((s['bbox'][0],s['origin'][1]),title,fontname='hebo',fontsize=8,color=(.1,.1,.1))
            placements.append({'letter':letter,'x':min(columns,key=lambda x:abs(x-s['bbox'][0]))-x0+3,
                               'y_top':s['bbox'][1]-14+18,
                               'old_heading':s['text'],'old_bbox':list(s['bbox'])})
        # A small top margin allows letters to sit above the full panel heading.
        final=fitz.open(); out=final.new_page(width=page.rect.width,height=page.rect.height+18)
        out.insert_font(fontname='hebo')
        out.show_pdf_page(fitz.Rect(0,18,page.rect.width,page.rect.height+18),doc,0)
        f=fitz.Font('hebo')
        row_groups=[]
        for p in placements:
            group=next((g for g in row_groups if abs(g[0]['y_top']-p['y_top'])<.3),None)
            if group is None:row_groups.append([p])
            else:group.append(p)
        existing_text=spans(out)
        for row,group in enumerate(row_groups):
            top=2 if row==0 else min(p['y_top'] for p in group)
            while True:
                collision=False
                for p in group:
                    box=fitz.Rect(p['x'],top,p['x']+f.text_length(p['letter'],fontsize=9),top+(f.ascender-f.descender)*9)
                    for s in existing_text:
                        inter=box & fitz.Rect(s['bbox'])
                        if not inter.is_empty and inter.width>.4 and inter.height>.4:collision=True
                if not collision:break
                top-=2
                assert top>=0,(name,'no clean panel-letter row')
            for p in group:p['y_top']=top
        for p in placements:
            baseline=p['y_top']+f.ascender*9
            out.insert_text((p['x'],baseline),p['letter'],fontname='hebo',fontsize=9,color=(.05,.05,.05))
        final.set_metadata({'title':Path(name).stem,'author':'','creator':'Vector presentation revision'})
        target=WORK/name
        final.save(target,garbage=4,deflate=True)
        final.close();doc.close()
        final=fitz.open(target);out=final[0]
        text=spans(out)
        checks=[]
        for p in placements:
            new=next(s for s in text if s['text']==p['letter'] and abs(s['bbox'][0]-p['x'])<.05 and abs(s['size']-9)<.01)
            box=fitz.Rect(new['bbox'])
            collisions=[]
            for s in text:
                if s is new:continue
                intersection=box & fitz.Rect(s['bbox'])
                if not intersection.is_empty and intersection.width>.4 and intersection.height>.4:collisions.append(s['text'])
            assert not collisions,(name,p['letter'],collisions)
            assert box.y1 < p['old_bbox'][1]+18-1,(name,'letter not above heading')
            assert box.x1 < p['old_bbox'][0]+1,(name,'letter not left of heading')
            checks.append({'letter':p['letter'],'bbox':list(box),'above_and_left':True,'text_overlap':False})
        # Equal rows/columns in the originals retain equal letter anchors.
        for a in checks:
            for b in checks:
                oa=next(p for p in placements if p['letter']==a['letter'])
                ob=next(p for p in placements if p['letter']==b['letter'])
                if abs(oa['old_bbox'][1]-ob['old_bbox'][1])<.2:assert abs(a['bbox'][1]-b['bbox'][1])<.2
        out.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(PREVIEW/(Path(name).stem+'.png'))
        REPORT['figures'].append({'name':name,'changed':True,'source_sha256':digest,'sha256':SHA(target),
                                  'vector_paths_unchanged_before_translation':True,'added_top_margin_pt':18,
                                  'panels':checks,'size_pt':list(out.rect)})


def save_new(fig,stem,anchors,details):
    for letter,x,y in anchors:
        fig.text(x,y,letter,fontsize=9,fontweight='bold',ha='left',va='bottom')
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    boxes=[]
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text():continue
        b=t.get_window_extent(renderer)
        if b.width and b.height:
            # Only count visible ticks (Matplotlib also creates off-axis ticks).
            if b.x1<0 or b.x0>fig.bbox.x1 or b.y1<0 or b.y0>fig.bbox.y1:continue
            assert b.x0>=-.5 and b.y0>=-.5 and b.x1<=fig.bbox.x1+.5 and b.y1<=fig.bbox.y1+.5,(stem,t.get_text(),list(b.bounds))
            boxes.append((t.get_text(),b))
    collisions=[]
    for i,(a,x) in enumerate(boxes):
        for b,y in boxes[i+1:]:
            if min(x.x1,y.x1)-max(x.x0,y.x0)>1.5 and min(x.y1,y.y1)-max(x.y0,y.y0)>1.5:collisions.append([a,b])
    assert not collisions,(stem,collisions)
    dest=WORK/'figures'/(stem+'.pdf')
    fig.savefig(dest,metadata={'Author':'','Creator':'Matplotlib','CreationDate':None,'ModDate':None})
    fig.savefig(PREVIEW/(stem+'.png'),dpi=200)
    REPORT['figures'].append({'name':'figures/'+dest.name,'changed':True,'new':True,'sha256':SHA(dest),
                             'text_overlap':False,'outside_text':False,'panel_anchors':anchors,**details})
    plt.close(fig)


def method():
    fig=plt.figure(figsize=(5.5,1.58))
    ax=fig.add_axes([.035,.08,.49,.69]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    def box(x,y,w,h,label,color='#F1F4F6',fs=8):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.025',fc=color,ec='#66727B',lw=.65))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fs)
    def arrow(a,b,color='#46525C',style='-'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,lw=.85,color=color,linestyle=style))
    for x,t in [(.025,r'$h_{\ell-1}$'),(.385,r'$h_\ell$'),(.80,r'$\hat y$')]:box(x,.62,.15,.20,t)
    arrow((.19,.72),(.365,.72));arrow((.55,.72),(.78,.72))
    ax.text(.275,.86,r'$W_\ell$',ha='center',fontsize=8)
    ax.text(.665,.84,r'$\cdots$',ha='center',fontsize=9)
    box(.80,.05,.15,.20,r'$e$',color='#FBEEDC')
    box(.385,.05,.15,.20,r'$\delta_\ell$',color='#FBEEDC')
    arrow((.875,.60),(.875,.28));arrow((.78,.15),(.555,.15),'#C07826','--')
    ax.text(.667,.31,r'fixed $B_\ell$',ha='center',fontsize=7.3)
    arrow((.46,.60),(.46,.28),'#C07826')
    ax.text(.35,.425,'local gate',ha='right',va='center',fontsize=7.2)
    fig.text(.06,.83,'Fixed feedback',fontsize=8,fontweight='bold')
    right=fig.add_axes([.61,.08,.36,.69]);right.set_xlim(0,1);right.set_ylim(0,1);right.axis('off')
    for y,label,expression,color in [(.79,'A',r'$G P_A$',COLORS['ndfa']),(.48,'E',r'$P_E G$',COLORS['endfa']),(.17,'K',r'$P_E G P_A$',COLORS['kndfa'])]:
        right.text(.04,y,label,ha='left',va='center',fontweight='bold',fontsize=8,color=color)
        right.text(.29,y,expression,ha='left',va='center',fontsize=10)
    fig.text(.635,.83,'Conditioned update',fontsize=8,fontweight='bold')
    save_new(fig,'ndfa_method_schematic_20260915',[('A',.006,.90),('B',.565,.90)],
             {'scope':'Conceptual DFA broadcast and A/E/K update, without claiming synaptic locality or unknown-noise identification.'})


def final_results():
    p=SUMMARY
    s=json.loads(p.read_text());assert s['accepted'] and len(s['contrasts'])==45
    fig=plt.figure(figsize=(5.5,2.45))
    axes=[fig.add_axes([.235,.205,.255,.595]),fig.add_axes([.735,.205,.245,.595])]
    designs=[('optimizer','dfa','SGD: A − DFA'),('optimizer','bp','SGD: A − BP'),
             ('forward','dfa','FD study: A − DFA'),('forward','fd_dfa','FD study: A − FD'),('forward','bp','FD study: A − BP')]
    rows=[next(r for r in s['contrasts'] if (r['cohort'],r['first'],r['second'])==(c,'ndfa',other)) for c,other,_ in designs]
    selected=[]
    for i,r in enumerate(rows):
        v=r['accuracy'];color=COLORS['ndfa'] if r['second']!='bp' else COLORS['bp']
        values=np.array(v['paired_values']);y=i+(np.arange(8)-3.5)*.029
        axes[0].scatter(values,y,s=6,color=color,alpha=.23,linewidths=0,zorder=2)
        axes[0].errorbar(v['mean'],i,xerr=[[v['mean']-v['ci95'][0]],[v['ci95'][1]-v['mean']]],
                         fmt='D' if r['primary_accuracy'] else 'o',color=color,ms=3.4,lw=1.15,capsize=2,zorder=3)
        selected.append(r)
    axes[0].set_yticks(range(5),[x[2] for x in designs]);axes[0].set_ylim(4.55,-.55)
    axes[0].set_xlim(-3.6,2.7);axes[0].set_xticks([-3,0,2]);axes[0].set_title('Matched work',loc='left',fontsize=8,fontweight='bold',pad=7)
    axes[0].axhline(1.5,color='.87',lw=.65)
    cells=[(10000,1024),(48000,1024),(10000,2048),(48000,2048)]
    for i,(n,w) in enumerate(cells):
        original=next(r for r in s['contrasts'] if (r['cohort'],r['first'],r['second'],r['n_train'],r['width'])==('factor','ndfa','kndfa',n,w))
        r=copy.deepcopy(original)
        r.update(first='kndfa',second='ndfa',source_orientation_reversed=True)
        for metric in ['accuracy','cross_entropy']:
            v=r[metric];v['mean']=-v['mean'];v['paired_values']=[-x for x in v['paired_values']]
            v['ci95']=[-v['ci95'][1],-v['ci95'][0]];v['positive'],v['negative']=v['negative'],v['positive']
        v=r['accuracy'];values=np.array(v['paired_values'])
        axes[1].scatter(values,i+(np.arange(8)-3.5)*.029,s=6,color=COLORS['kndfa'],alpha=.23,linewidths=0,zorder=2)
        axes[1].errorbar(v['mean'],i,xerr=[[v['mean']-v['ci95'][0]],[v['ci95'][1]-v['mean']]],fmt='o',color=COLORS['kndfa'],ms=3.4,lw=1.15,capsize=2,zorder=3)
        selected.append(r)
    axes[1].set_yticks(range(4),['10k · 3.68M','48k · 3.68M','10k · 8.41M','48k · 8.41M']);axes[1].set_ylim(3.55,-.55)
    axes[1].set_xlim(-3.6,2.7);axes[1].set_xticks([-3,0,2]);axes[1].set_title('Adding K to A',loc='left',fontsize=8,fontweight='bold',pad=7)
    # K is the combined rule; the label explicitly identifies the comparison.
    axes[1].set_title('K − A',loc='left',fontsize=8,fontweight='bold',pad=7)
    for ax in axes:
        ax.axvline(0,color='.55',ls=':',lw=.8,zorder=1);ax.set_xlabel('Test accuracy gain (pp)',labelpad=4)
        ax.tick_params(length=2.5,pad=3);ax.grid(axis='x',color='.93',lw=.5)
    # Every plotted seed must remain inside the visible axis limits.
    for r in selected:
        assert min(r['accuracy']['paired_values'])>-3.6 and max(r['accuracy']['paired_values'])<2.7
    save_new(fig,'ndfa_final_test_summary_20260915',[('A',.006,.93),('B',.535,.93)],
             {'source_sha256':SHA(p),'plotted_contrasts':selected,'paired_seeds_each':8,
              'panel_plot_y_bounds':[.205,.800],'no_cohort_pooling':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-source',type=Path,default=Path(__file__).parent/'original_source')
    parser.add_argument('--workspace',type=Path,default=WORK)
    parser.add_argument('--preview',type=Path,default=PREVIEW)
    parser.add_argument('--summary',type=Path,default=SUMMARY)
    parser.add_argument('--report',type=Path,default=BASE/'figure_revision.json')
    args=parser.parse_args()
    SOURCE=args.original_source.resolve();WORK=args.workspace.resolve();PREVIEW=args.preview.resolve();SUMMARY=args.summary.resolve()
    PREVIEW.mkdir(parents=True,exist_ok=True);(WORK/'figures').mkdir(parents=True,exist_ok=True)
    REPORT['source_manifest_sha256']=SHA(SOURCE/'manifest.json')
    existing_figures();method();final_results()
    REPORT['generator_sha256']=SHA(__file__)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(REPORT,indent=2)+'\n')
    print(json.dumps({'figures':len(REPORT['figures']),'changed':sum(r['changed'] for r in REPORT['figures']),
                      'new':sum(r.get('new',False) for r in REPORT['figures'])}))
