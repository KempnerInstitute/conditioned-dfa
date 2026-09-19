"""Record figure text bounds and render final pages for visual inspection."""
from pathlib import Path
import json,re
import fitz
ROOT=Path(__file__).resolve().parents[2];PAPER=ROOT/'drafts/Info-DFA';OUT=PAPER/'build/comments_revision_20260918/visual_review';OUT.mkdir(parents=True,exist_ok=True)
def main():
    text=(PAPER/'paper_body.tex').read_text()+(PAPER/'supplement.tex').read_text();rows=[]
    for name in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text):
        doc=fitz.open(PAPER/'figures'/name);page=doc[0];bounds=page.rect;outside=[]
        for t in page.get_texttrace():
            box=fitz.Rect(t['bbox']);word=''.join(chr(c[0]) for c in t['chars'])
            if box.x0<-.8 or box.y0<-.8 or box.x1>bounds.x1+.8 or box.y1>bounds.y1+.8:outside.append({'text':word,'box':list(box)})
        rows.append({'figure':name,'width_pt':bounds.width,'height_pt':bounds.height,'outside_text':outside})
        page.get_pixmap(matrix=fitz.Matrix(1.8,1.8),alpha=False).save(OUT/(Path(name).stem+'.png'))
    p=fitz.open(PAPER/'build/comments_revision_20260918/release_07/conditioned_dfa_iclr.pdf')
    for i in range(9):p[i].get_pixmap(matrix=fitz.Matrix(1.15,1.15),alpha=False).save(OUT/f'main_{i+1:02d}.png')
    # PDF contact sheets preserve vector content and allow full-page review.
    sheets=fitz.open()
    for start in range(0,len(p),8):
        sheet=sheets.new_page(width=612,height=792)
        for j,index in enumerate(range(start,min(start+8,len(p)))):
            col,row=j%2,j//2;rect=fitz.Rect(col*306,row*198,(col+1)*306,(row+1)*198);sheet.show_pdf_page(rect,p,index,keep_proportion=True)
    sheets.save(OUT/'page_contact_sheets.pdf')
    (OUT/'figure_text_bounds.json').write_text(json.dumps({'figures':rows,'clipped_text_count':sum(len(x['outside_text']) for x in rows),'scope':'Text bounds plus page renders; semantic overlaps require visual review.'},indent=2)+'\n')
    print(json.dumps([x for x in rows if x['outside_text']],indent=2));print('Rendered',len(rows),'figures and',len(p),'pages')
if __name__=='__main__':main()
