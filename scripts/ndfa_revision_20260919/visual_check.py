"""Check vector text bounds and render the curated manuscript for review."""
from pathlib import Path
import argparse,json,re
import fitz
ROOT=Path(__file__).resolve().parents[2]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--paper-build',type=Path,required=True)
    a=parser.parse_args();paper=ROOT/'drafts/Info-DFA';out=a.paper_build/'visual';out.mkdir(exist_ok=True)
    source=(paper/'paper_body.tex').read_text()+(paper/'supplement.tex').read_text();rows=[]
    for name in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',source):
        doc=fitz.open(paper/'figures'/name);page=doc[0];bounds=page.rect;outside=[]
        for t in page.get_texttrace():
            box=fitz.Rect(t['bbox'])
            if box.x0<-.8 or box.y0<-.8 or box.x1>bounds.x1+.8 or box.y1>bounds.y1+.8:
                outside.append({'text':''.join(chr(c[0]) for c in t['chars']),'box':list(box)})
        rows.append({'figure':name,'width_pt':bounds.width,'height_pt':bounds.height,'outside_text':outside})
        page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(out/(Path(name).stem+'.png'))
    manuscript=fitz.open(a.paper_build/'conditioned_dfa_iclr.pdf')
    for start in range(0,len(manuscript),4):
        sheet=fitz.open();page=sheet.new_page(width=1224,height=1584)
        for j,index in enumerate(range(start,min(start+4,len(manuscript)))):
            col,row=j%2,j//2;rect=fitz.Rect(col*612,row*792,(col+1)*612,(row+1)*792)
            page.show_pdf_page(rect,manuscript,index)
        page.get_pixmap(alpha=False).save(out/f'contact_{start+1:02d}.png')
    for i in range(9):manuscript[i].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(out/f'main_{i+1:02d}.png')
    result={'figures':rows,'clipped_text_count':sum(len(x['outside_text']) for x in rows),
            'scope':'Text bounds and page renders; visual review assesses semantic overlap.'}
    (out/'figure_text_bounds.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['clipped_text_count']==0,result
    print('No clipped figure text;',len(rows),'figures;',len(manuscript),'pages rendered.')
if __name__=='__main__':main()
